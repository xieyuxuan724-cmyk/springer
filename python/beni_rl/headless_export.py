# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Export a BENI training checkpoint to ONNX and MNN on CPU, without a simulator."""

from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml
from rsl_rl.models import RNNModel
from tensordict import TensorDict


class AgentConfigLoader(yaml.SafeLoader):
    """Read MJLab's tuple tags without allowing arbitrary Python objects."""


AgentConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node)),
)


def resolve_checkpoint(run_dir: Path | None, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        path = checkpoint.expanduser()
        if run_dir is not None and not path.is_absolute():
            path = run_dir.expanduser() / path
    elif run_dir is not None:
        candidates = [
            path
            for path in run_dir.expanduser().glob("model_*.pt")
            if path.is_file() and re.fullmatch(r"model_\d+\.pt", path.name)
        ]
        if not candidates:
            raise FileNotFoundError(f"No model_<iteration>.pt checkpoints found in {run_dir}")
        path = max(candidates, key=lambda item: int(item.stem.removeprefix("model_")))
    else:
        raise ValueError("Provide --checkpoint /path/to/model.pt or --run_dir /path/to/run")
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path.resolve()


def load_actor(checkpoint: Path, agent_config: Path) -> RNNModel:
    """Restore the recurrent actor and its observation normalization for export."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or "actor_state_dict" not in saved:
        raise ValueError("Expected an BENI MJLab checkpoint with 'actor_state_dict'")
    with agent_config.open(encoding="utf-8") as stream:
        config = yaml.load(stream, Loader=AgentConfigLoader)
    if not isinstance(config, dict) or not isinstance(config.get("actor"), dict):
        raise ValueError(f"Missing actor configuration in {agent_config}")
    actor_cfg = config["actor"].copy()
    if actor_cfg.pop("class_name", None) != "RNNModel":
        raise ValueError("Only the BENI RNNModel actor is supported")
    if actor_cfg.pop("cnn_cfg", None) is not None:
        raise ValueError("The BENI actor must not contain a CNN")
    if (
        actor_cfg.get("rnn_type") != "gru"
        or actor_cfg.get("rnn_hidden_dim") != 512
        or actor_cfg.get("rnn_num_layers") != 1
    ):
        raise ValueError("The BENI deployment interface requires a single-layer GRU-512")
    distribution = actor_cfg.get("distribution_cfg")
    if not isinstance(distribution, dict) or distribution.get("class_name") != "GaussianDistribution":
        raise ValueError("The BENI actor must use GaussianDistribution")

    observation_count = saved['actor_state_dict']['rnn.rnn.weight_ih_l0'].shape[1]
    if observation_count not in (40, 42):
        raise ValueError(f'Unsupported BENI observation count: {observation_count}')
    actor = RNNModel(
        obs=TensorDict({"actor": torch.zeros(1, observation_count)}, batch_size=[1]),
        obs_groups={"actor": ["actor"]},
        obs_set="actor",
        output_dim=7,
        **actor_cfg,
    )
    # Training keeps 40 inputs; retain export support for existing 42-input checkpoints.
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    return actor.eval()


def export_onnx(actor: RNNModel, output_path: Path) -> None:
    model = actor.as_onnx().cpu().eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        model.get_dummy_inputs(),
        str(output_path),
        export_params=True,
        opset_version=18,
        input_names=model.input_names,
        output_names=model.output_names,
        dynamic_axes={},
        dynamo=False,
    )


def convert_to_mnn(onnx_path: Path, mnn_path: Path, fp16: bool) -> None:
    if importlib.util.find_spec("MNN") is None:
        raise FileNotFoundError(
            "MNN is missing. Run: uv run --group export python -m beni_rl.headless_export ... "
            "(use --skip_mnn for ONNX only)."
        )
    # Some converter versions return zero without producing a model. Convert into
    # a fresh directory so an older policy.mnn cannot be mistaken for success.
    with tempfile.TemporaryDirectory(prefix=".mnn-export-", dir=mnn_path.parent) as directory:
        temporary_model = Path(directory) / "policy.mnn"
        command = [
            sys.executable,
            "-c",
            # The CLI wrapper imports a logger that runs system pip. Use the
            # same converter binding from the uv-installed wheel directly.
            "import sys; import _tools; _tools.mnnconvert(sys.argv)",
            "-f",
            "ONNX",
            "--modelFile",
            str(onnx_path),
            "--MNNModel",
            str(temporary_model),
            "--optimizeLevel",
            "1",
            "--optimizePrefer",
            "2",
            "--bizCode",
            "OpenBeni",
        ]
        if fp16:
            command.append("--fp16")
        subprocess.run(command, check=True)
        if not temporary_model.is_file() or temporary_model.stat().st_size == 0:
            raise RuntimeError("MNN conversion did not produce a nonempty model")
        temporary_model.replace(mnn_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", "--run-dir", type=Path, help="Training run; defaults to its latest checkpoint")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint path, or filename relative to --run_dir")
    parser.add_argument(
        "--agent_config",
        "--agent-config",
        type=Path,
        help="Default: <checkpoint-dir>/params/agent.yaml",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=Path,
        help="Default: <checkpoint-dir>/exported/<checkpoint-stem>",
    )
    parser.add_argument("--skip_mnn", "--skip-mnn", action="store_true", help="Export ONNX only")
    parser.add_argument("--fp32", action="store_true", help="Store MNN weights in FP32 (default: FP16)")
    args = parser.parse_args()
    if args.run_dir is None and args.checkpoint is None:
        parser.error("--run_dir or --checkpoint is required")

    checkpoint = resolve_checkpoint(args.run_dir, args.checkpoint)
    agent_config = args.agent_config.expanduser() if args.agent_config else checkpoint.parent / "params" / "agent.yaml"
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir else checkpoint.parent / "exported" / checkpoint.stem
    )
    actor = load_actor(checkpoint, agent_config)
    print(f"Checkpoint: {checkpoint}", flush=True)
    onnx_path = output_dir / "policy.onnx"
    export_onnx(actor, onnx_path)
    print(f"ONNX: {onnx_path}", flush=True)
    if not args.skip_mnn:
        mnn_path = output_dir / "policy.mnn"
        convert_to_mnn(onnx_path, mnn_path, fp16=not args.fp32)
        print(f"MNN: {mnn_path}", flush=True)
    width = actor.rnn.rnn.input_size
    print(f"Inputs: obs [1,{width}], h_in [1,1,512]; outputs: actions [1,7], h_out [1,1,512]")


if __name__ == "__main__":
    main()
