#!/usr/bin/env python3
"""Train the BENI upper policy using the repository's RSL-RL PPO stack."""
import argparse
from datetime import datetime
import importlib.metadata
import json
import math
from pathlib import Path
import time
import torch
from beni_avoidance.common import Config, ROOT, fingerprint
from beni_avoidance.env import AvoidanceEnv
from beni_avoidance.training import Adapter,Runner,BudgetExpired,runner_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['native','warp'],default='warp')
    p.add_argument('--num-envs',type=int,default=32)
    p.add_argument('--iterations',type=int,default=2000)
    p.add_argument('--minutes',type=float,default=80)
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--bottom',type=Path,default=ROOT/'assets/models/policy.pt')
    p.add_argument('--run-dir',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--validation-interval',type=int,default=50,help='Native validation every N iteration indices; 0 disables')
    p.add_argument('--validation-episodes',type=int,default=12,help='Balanced fixed-speed validation episodes, multiple of 3')
    p.add_argument('--validation-zero-episodes',type=int,default=3)
    p.add_argument('--validation-seconds',type=float,default=120,help='Wall-clock limit per validation round, inside training budget')
    a=p.parse_args()
    if min(a.num_envs,a.iterations,a.steps)<=0 or a.minutes<=0:p.error('Counts and budget must be positive')
    if (a.validation_interval<0 or a.validation_episodes<3 or a.validation_episodes%3
        or a.validation_zero_episodes<1 or a.validation_seconds<=0):p.error('Invalid validation settings')
    validation=dict(interval=a.validation_interval,episodes=a.validation_episodes,
                    zero_episodes=a.validation_zero_episodes,seconds=a.validation_seconds)
    torch.set_num_threads(1);torch.manual_seed(a.seed)
    cfg=Config(seed=a.seed,bottom=str(a.bottom.resolve()))
    saved=None
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu',weights_only=True)
        cfg=Config(**saved['infos']['avoidance_config']);cfg.bottom=str(a.bottom.resolve())
        if fingerprint(cfg.bottom)!=saved['infos']['bottom_sha256']:raise ValueError('Bottom policy mismatch')
    run=a.run_dir or ROOT/'logs/rsl_rl/beni_avoidance'/datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run.mkdir(parents=True,exist_ok=False)
    init=time.monotonic();env=AvoidanceEnv(a.num_envs,a.backend,cfg)
    tc=runner_config(cfg.seed,a.steps)
    if saved:tc=saved['infos']['runner_config']
    if a.validation_interval:tc['save_interval']=math.gcd(tc['save_interval'],a.validation_interval)
    adapter=Adapter(env,tc['num_steps_per_env'],log=run/'episodes.jsonl')
    runner=Runner(adapter,tc,str(run),env.device,validation=validation)
    if saved:
        runner.alg.load(saved,None,True);runner.current_learning_iteration=saved['iter']+1
        adapter.total_steps=saved['infos']['collected_upper_steps']
        env.rng.bit_generator.state=saved['infos']['environment_rng']
        torch.set_rng_state(saved['infos']['torch_rng'])
        if torch.cuda.is_available() and saved['infos']['cuda_rng']:torch.cuda.set_rng_state_all(saved['infos']['cuda_rng'])
    manifest=dict(config=cfg.dict(),bottom_sha256=fingerprint(cfg.bottom),backend=a.backend,num_envs=a.num_envs,
                  runner=runner.original_config,validation=validation,versions={k:importlib.metadata.version(k) for k in ('mujoco','mujoco-warp','torch','rsl-rl-lib')},
                  initialization_seconds=time.monotonic()-init)
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str))
    started=time.monotonic();adapter.deadline=started+a.minutes*60
    result='iterations_complete'
    try:runner.learn(a.iterations)
    except (BudgetExpired,KeyboardInterrupt) as exc:result=type(exc).__name__
    except Exception:
        result='failed'
        raise
    finally:
        runner.save(str(run/'model_final.pt'))
        runner.logger.stop_logging_writer()
        (run/'result.json').write_text(json.dumps(dict(status=result,seconds=time.monotonic()-started,
            iteration=runner.current_learning_iteration,collected_steps=adapter.total_steps),indent=2))
    print(f'Run saved to {run}')


if __name__=='__main__':main()
