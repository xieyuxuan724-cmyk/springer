# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Measure recovery before advancing the training curriculum to Jump."""
from copy import deepcopy
from dataclasses import dataclass, replace
import random
import math
import hashlib

import numpy as np
import torch

from . import ASSETS_DIR
from .recovery import restore_recorded_state
from .recovery_curriculum import KINDS
from .recovery_bank import SUBGROUPS, bank_digest, load_recovery_bank, recovery_subgroups

RECOVERY_PROTOCOL = 'stratified_training_reset_and_termination'


@dataclass
class RecoveryGateCfg:
    check_interval: int = 500
    performance_interval: int = 1000
    heldout_seed: int = 31415
    heldout_bank: str = str(ASSETS_DIR / "motions" / "recovery_contact_states_heldout.npz")
    trials: int = 128
    min_valid_trials: int = 64
    success_rate: float = .8
    consecutive_passes: int = 2
    """Full recovery confirmations required for Jump; each supine increment needs one pass."""
    scenarios: tuple[str, ...] = KINDS[1:]
    subgroups: tuple[str, ...] = ('base_front_free', 'base_front_head', 'base_rear_free', 'base_rear_head',
                                 'legs_folded', 'legs_asymmetric', 'legs_extended')
    deadline_s: float = 6.
    hold_s: float = 1.
    supine_start_iteration: int = 1500
    """Introduce grounded supine trials, then check after each check_interval updates."""
    supine_initial_grounding_probability: float = .2
    supine_grounding_step: float = .4
    """Add this fraction after each successful check, starting from the initial probability."""
    supine_success_rate: float = .7
    force_jump_after_updates: int | None = 9000
    """Enter Jump after this many completed updates even without recovery qualification; None disables the limit."""

    def validate(self):
        if (self.check_interval < 1 or self.performance_interval < 1 or self.heldout_seed < 0 or self.trials < self.min_valid_trials or self.min_valid_trials < 1
                or not 0 < self.success_rate <= 1 or self.consecutive_passes < 1
                or not math.isfinite(self.deadline_s) or not math.isfinite(self.hold_s)
                or self.deadline_s <= 0 or self.hold_s <= 0 or self.supine_start_iteration < 0
                or not 0 < self.supine_initial_grounding_probability <= 1
                or not 0 < self.supine_grounding_step <= 1
                or not 0 < self.supine_success_rate <= 1
                or (self.force_jump_after_updates is not None
                    and (not isinstance(self.force_jump_after_updates, int) or self.force_jump_after_updates < 1))):
            raise ValueError('Invalid recovery curriculum thresholds')
        if len(set(self.subgroups)) != len(self.subgroups) or set(self.subgroups)-set(SUBGROUPS):
            raise ValueError('Invalid recovery subgroups')
        if not self.scenarios or len(set(self.scenarios)) != len(self.scenarios) or set(self.scenarios)-set(KINDS[1:]):
            raise ValueError(f'Recovery scenarios must be distinct members of {KINDS[1:]}')


def recovery_gate_result(stable, valid, dt, cfg, terminated=None):
    """Count only valid releases with a fully observed continuous standing hold."""
    if terminated is not None:
        # A later recovery cannot undo a failure that would have ended training's episode.
        stable = stable & ~terminated.to(torch.long).cummax(dim=0).values.bool()
    hold = max(1, round(cfg.hold_s/dt))
    last_start = round(cfg.deadline_s/dt)
    run = torch.zeros_like(valid, dtype=torch.long)
    recovered = torch.zeros_like(valid)
    onset = torch.full(valid.shape, float("inf"), device=valid.device)
    # Include a hold beginning exactly at the deadline, but never an incomplete hold.
    for step, frame in enumerate(stable[:last_start+hold]):
        run = torch.where(frame, run+1, 0)
        first = (run >= hold) & ~recovered
        onset[first] = (step-hold+1)*dt
        recovered |= run >= hold
    results = {}
    for index, name in enumerate(cfg.scenarios):
        sl = slice(index*cfg.trials, (index+1)*cfg.trials)
        count = int(valid[sl].sum())
        successes = int((recovered[sl] & valid[sl]).sum())
        results[name] = dict(valid=count, invalid=cfg.trials-count, recovered=successes,
                             success_rate=successes/count if count else None,
                             passed=count >= cfg.min_valid_trials and successes >= cfg.success_rate*count)
        times = onset[sl][valid[sl] & recovered[sl]]
        results[name]['recovery_time_p90_s'] = float(torch.quantile(times, .9)) if len(times) else None
        for seconds in (1,3,6):
            results[name][f'recovered_by_{seconds}s_rate'] = (int((valid[sl] & (onset[sl] <= seconds+1e-6)).sum())/count
                                                            if count else None)
        results[name]['trials'] = [dict(valid=bool(v), recovered=bool(r and v),
                                              recovery_s=float(t) if r and v else None)
                                  for v,r,t in zip(valid[sl].cpu(), recovered[sl].cpu(), onset[sl].cpu())]
    return dict(passed=all(row['passed'] for row in results.values()), scenarios=results)


def recovery_initial_masks(env, sample):
    """Classify the released state, including upright-looking support failures."""
    sensor = env.scene['body_ground_contact']
    body_now = sensor.data.force.norm(dim=-1) > .1
    def contact(suffix):
        return body_now[:, [i for i, name in enumerate(sensor.primary_names) if name.endswith(suffix)]].any(-1)
    wheels = sample['contacts'].sum(-1)
    # At reset there are no past physics substeps; classify current forces, not cleared history.
    body = body_now.any(-1)
    fallen = body & ((sample['tilt'] >= 70) | (sample['height'] < .065))
    gravity_x = env.scene['robot'].data.projected_gravity_b[:,0]
    masks = dict(supine=fallen & (gravity_x < -.5),
                 prone=fallen & (gravity_x > .5),
                 upside_down=fallen & (sample['tilt'] >= 150))
    masks.update(
        belly_support=(sample['tilt'] < 60) & contact('base_link') & (wheels == 0),
        leg_support=(sample['tilt'] < 60) & contact(('1_Link', '2_Link')) & ~contact('base_link') & (wheels == 0),
        head_support=contact('head') & (wheels < 2),
        one_wheel=wheels == 1,
        mixed_support=(wheels == 2) & body)
    return masks


def valid_recovery_initials(env, sample, cfg):
    masks = recovery_initial_masks(env, sample)
    tags = recovery_subgroups(env, sample)
    physical = torch.stack(tuple(masks.values())).any(0)
    masks.update({name: tags[:,i] & physical for i,name in enumerate(SUBGROUPS)})
    for name in cfg.scenarios:
        if '/' in name:
            kind,subgroup=name.split('/')
            masks[name]=masks[kind] & masks[subgroup]
    return torch.cat([masks[name][i*cfg.trials:(i+1)*cfg.trials] for i, name in enumerate(cfg.scenarios)])


@torch.inference_mode()
def check_recovery(env_cfg, actor, cfg, seed, device, bank_path=None):
    """Check deterministic actions on a fixed training-bank subset, without resets.

    Uses a separate simulator and copied inference model. Training RNG streams,
    GRU state, normalization and episodes are preserved. This curriculum check
    does not replace independent held-out performance evaluation.
    """
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import EventTermCfg, TerminationManager
    from .jump import JumpMeasurement

    cfg.validate()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    env = None
    with torch.random.fork_rng(devices=[torch.device(device)] if torch.device(device).type == 'cuda' else []):
        try:
            check_cfg = deepcopy(env_cfg)
            check_cfg.seed = seed
            leg_strata = tuple(f'{kind}/{tag}' for kind in cfg.scenarios if kind in KINDS[1:4]
                               for tag in cfg.subgroups if tag.startswith('legs_'))
            names = (*cfg.scenarios, *cfg.subgroups, *leg_strata)
            assessment_cfg = replace(cfg, scenarios=names, subgroups=())
            check_cfg.scene.num_envs = cfg.trials*len(names)
            check_cfg.terminations = {}; check_cfg.curriculum = {}; check_cfg.rewards = {}; check_cfg.metrics = {}
            check_cfg.recorders.pop('recovery_sampling', None)
            check_cfg.auto_reset = False
            check_cfg.events = {k: v for k, v in check_cfg.events.items() if k in ('reset_base', 'reset_joints')}
            check_cfg.events['reset_base'].params.update(fallen_probability=0., pose_range={})
            bank_path = bank_path or env_cfg.events['reset_curriculum'].params['bank_path']
            bank = load_recovery_bank(bank_path)
            positions = np.concatenate([bank[k+'_qpos'] for k in KINDS[1:]])
            velocities = np.concatenate([bank[k+'_qvel'] for k in KINDS[1:]])
            labels = np.concatenate([bank.get(k+'_tags', np.zeros((len(bank[k+'_qpos']),len(SUBGROUPS)), bool))
                                     for k in KINDS[1:]])
            origin_kind = np.concatenate([np.full(len(bank[k+'_qpos']), k) for k in KINDS[1:]])
            origin_index = np.concatenate([np.arange(len(bank[k+'_qpos'])) for k in KINDS[1:]])
            # A state shared by two broad classes must not count twice in a subgroup's denominator.
            _, distinct = np.unique(np.concatenate((positions,velocities),axis=1), axis=0, return_index=True)
            unique = np.zeros(len(positions),bool); unique[distinct] = True
            selections, available, choices = [], [], {}
            for name in names:
                if '/' in name:
                    kind,tag=name.split('/')
                    mask=(origin_kind==kind) & labels[:,SUBGROUPS.index(tag)]
                else:
                    mask = origin_kind == name if name in KINDS else labels[:,SUBGROUPS.index(name)] & unique
                candidates = np.flatnonzero(mask)
                if len(candidates):
                    _, own_distinct = np.unique(np.concatenate((positions[candidates],velocities[candidates]),axis=1),
                                                axis=0,return_index=True)
                    candidates = candidates[own_distinct]
                name_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
                selected = np.random.default_rng(seed+name_seed).permutation(candidates)[:cfg.trials]
                count = len(selected)
                available.extend([True]*count + [False]*(cfg.trials-count))
                # Padding maintains a fixed simulator batch; padded releases are always invalid.
                selected = np.pad(selected,(0,cfg.trials-count))
                selections.extend(selected.tolist())
                choices[name] = origin_index[selected].tolist()
            selections = np.array(selections)
            qpos = torch.tensor(positions[selections], device=device)
            qvel = torch.tensor(velocities[selections], device=device)

            def reset_initials(env, env_ids):
                q = qpos[env_ids].clone()
                q[:, :3] += env.scene.env_origins[env_ids]
                restore_recorded_state(env, q, qvel[env_ids], env_ids)

            def record_termination(env, env_ids):
                # Step events run at the native termination sampling phase, before forward/sense.
                # Keep the trajectory running for observation, but record the original failure.
                terminated.copy_(termination_manager.compute())

            check_cfg.events['reset_curriculum'] = EventTermCfg(func=reset_initials, mode='reset')
            check_cfg.events['recovery_termination'] = EventTermCfg(func=record_termination, mode='step')
            for group in check_cfg.observations.values():
                group.enable_corruption = False
            check_cfg.commands['jump'].play = False
            twist = check_cfg.commands['twist']
            twist.resampling_time_range = (1e9, 1e9)
            twist.rel_standing_envs = 1.
            twist.heading_command = False
            twist.ranges.heading = None
            twist.pitch_range = twist.roll_range = (0., 0.)
            twist.height_range = (.11, .11)
            env = ManagerBasedRlEnv(check_cfg, device=device)
            termination_manager = TerminationManager(deepcopy(env_cfg.terminations), env)
            terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            # The native reset clears action/history buffers and senses the restored state at t=0.
            obs, _ = env.reset(seed=seed)
            count = env.num_envs
            measure = JumpMeasurement(env)
            initial = measure(env)
            valid = valid_recovery_initials(env, initial, assessment_cfg)
            valid &= torch.tensor(available, device=device)
            policy = actor.as_onnx().to(device).eval()
            hidden = torch.zeros(actor.rnn.rnn.num_layers, count, actor.rnn.rnn.hidden_size, device=device)
            stable = [initial['stable'] & ~initial['body_current']]
            terminations = [terminated.clone()]
            for _ in range(round((cfg.deadline_s+cfg.hold_s)/env.step_dt)):
                action, hidden, _ = policy(obs['actor'], hidden)
                obs, *_ = env.step(action)
                stable.append(measure(env)['stable'].clone())
                terminations.append(terminated.clone())
            result = recovery_gate_result(torch.stack(stable), valid, env.step_dt, assessment_cfg, torch.stack(terminations))
            for i,name in enumerate(names):
                for j,row in enumerate(result['scenarios'][name]['trials']):
                    index = selections[i*cfg.trials+j]
                    row.update(kind=str(origin_kind[index]), index=int(origin_index[index]),
                               subgroups=[key for key,yes in zip(SUBGROUPS,labels[index]) if yes])
            result.update(seed=seed, bank_path=str(bank_path), choices=choices, bank_sha256=bank_digest(bank_path),
                          deadline_s=cfg.deadline_s, hold_s=cfg.hold_s, auto_reset=False,
                          protocol=RECOVERY_PROTOCOL, release_s=0., termination_terms=termination_manager.active_terms)
            return result
        finally:
            if env is not None:
                env.close()
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
