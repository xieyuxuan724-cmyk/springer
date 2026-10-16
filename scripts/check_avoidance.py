#!/usr/bin/env python3
"""Bounded performance/bottom-policy smoke test; does not train a policy."""
import argparse,json,time
from pathlib import Path
import torch
from beni_avoidance.env import AvoidanceEnv,rule_action
from beni_avoidance.common import Config
from beni_avoidance.scene_cli import add_scene_arguments,apply_scene_arguments


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['native','warp'],default='native')
    p.add_argument('--num-envs',type=int,default=1)
    p.add_argument('--steps',type=int,default=30)
    p.add_argument('--count',type=int,default=0)
    p.add_argument('--speed',type=float,default=.3)
    p.add_argument('--yaw',type=float,default=None,help='Direct upper normalized yaw command for bottom tests')
    p.add_argument('--log',type=Path)
    add_scene_arguments(p)
    a=p.parse_args()
    cfg=Config()
    try:apply_scene_arguments(cfg,a)
    except ValueError as exc:p.error(str(exc))
    torch.set_num_threads(1)
    if a.num_envs<1 or a.steps<1 or not 0<=a.count<=cfg.max_obstacles:p.error('Invalid count or steps')
    if a.backend=='warp':torch.cuda.reset_peak_memory_stats()
    start=time.monotonic();e=AvoidanceEnv(a.num_envs,a.backend,cfg)
    e.reset(seeds=list(range(9_000_000,9_000_000+a.num_envs)),speeds=[a.speed]*a.num_envs,
            count=a.count,directions=[0.]*a.num_envs)
    initialized=time.monotonic();failures=[]
    for i in range(a.steps):
        action=rule_action(e)
        if a.yaw is not None:action[:,0]=1;action[:,1]=a.yaw
        _,_,done,info=e.step(action)
        if done.any():
            failures.extend([dict(step=i,environment=j,reason=int(info['reason'][j])) for j in done.nonzero().flatten().tolist()])
            break
    if a.backend=='warp':torch.cuda.synchronize()
    seconds=time.monotonic()-initialized
    report=dict(backend=a.backend,num_envs=a.num_envs,steps=i+1,initialization_s=initialized-start,
                simulation_s=(i+1)*.1,wall_s=seconds,upper_transitions_per_s=(i+1)*a.num_envs/seconds,
                positions=e.backend.qpos[:,:3].cpu().tolist(),failures=failures,
                peak_torch_allocated_mb=torch.cuda.max_memory_allocated()/2**20 if a.backend=='warp' else 0,
                note='Torch allocation excludes Warp allocations; inspect nvidia-smi for total device use.')
    print(json.dumps(report,indent=2))
    if a.log:a.log.parent.mkdir(parents=True,exist_ok=True);a.log.write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
