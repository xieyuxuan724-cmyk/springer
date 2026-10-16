#!/usr/bin/env python3
"""Fixed held-out test protocol; rule baseline and learned actor use identical seeds."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from beni_avoidance.common import Config,fingerprint
from beni_avoidance.env import AvoidanceEnv,rule_action
from beni_avoidance.training import Upper
from beni_avoidance.scene_cli import add_scene_arguments,apply_scene_arguments,standard_scene

from beni_avoidance.metrics import REASONS,write_evaluation


def evaluate(env,policy,episodes=100,seed_base=3_000_000,zero=False,deadline=None,obstacles=None):
    records=[];slots={};next_episode=0
    def assign(slot,index):
        seed=seed_base+index
        speed=0. if zero else [.1,.3,.5][index%3]
        angle=float(np.random.default_rng(seed).uniform(-np.pi,np.pi))
        count=obstacles if obstacles is not None else 3+seed%(env.cfg.max_obstacles-2)
        env.reset([slot],seeds=[seed],speeds=[speed],count=count,directions=[angle])
        slots[slot]=dict(index=index,seed=seed,speed=speed,count=count)
    for slot in range(min(env.n,episodes)):assign(slot,next_episode);next_episode+=1
    while slots:
        if deadline is not None and time.monotonic()>=deadline:raise TimeoutError("Validation budget exhausted")
        action=rule_action(env) if policy is None else policy(env.observation())
        _,_,done,info=env.step(action)
        for slot in list(slots):
            if bool(done[slot]):
                rec=slots.pop(slot)
                rec.update(reason=REASONS[int(info['reason'][slot])],success=bool(info['success'][slot]),
                    time_s=float(info['time'][slot]),progress_m=float(info['progress'][slot]),
                    min_clearance_m=float(info['clearance'][slot]),mean_heading_error_rad=float(info['heading_error'][slot]),
                    mean_speed_error_mps=float(info['speed_error'][slot]),final_drift_m=float(info['drift'][slot]),
                    max_drift_m=float(info['max_drift'][slot]),final_speed_mps=float(info['final_speed'][slot]))
                records.append(rec)
                print(f'episode {len(records)}/{episodes}: {rec["reason"]}, progress={rec["progress_m"]:.2f}',flush=True)
                if next_episode<episodes:assign(slot,next_episode);next_episode+=1
        # Finished slots are reset harmlessly; never allowed to diverge while other episodes finish.
        unused=[i for i in range(env.n) if i not in slots]
        if unused:env.reset(unused,seeds=[9_900_000+i for i in unused],speeds=[0.]*len(unused),count=0,directions=[0.]*len(unused))
    standard=standard_scene(env.cfg) and obstacles is None
    records.sort(key=lambda r:r['index'])
    rate=sum(r['success'] for r in records)/episodes
    grouped={str(v):dict(episodes=sum(r['speed']==v for r in records),
        success_rate=float(np.mean([r['success'] for r in records if r['speed']==v])))
        for v in sorted(set(r['speed'] for r in records))}
    return dict(stress_test=not standard,obstacle_count_override=obstacles,episodes=episodes,seed_base=seed_base,zero_command_test=zero,success_rate=rate,
                formal_protocol=standard and episodes==100 and seed_base==3_000_000 and not zero,
                acceptance_met=rate>=.9 if standard and episodes==100 and seed_base==3_000_000 and not zero else None,
                reason_counts={reason:sum(r['reason']==reason for r in records) for reason in REASONS.values()},
                by_speed=grouped,records=records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,help='Omit for the rule baseline')
    p.add_argument('--backend',choices=['native','warp'],default='native')
    p.add_argument('--num-envs',type=int,default=1)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--seed-base',type=int,default=3_000_000,help='Use 2000000 for validation, 3000000 only for final test')
    p.add_argument('--zero',action='store_true')
    p.add_argument('--bottom',type=Path)
    p.add_argument('--tensorboard-dir',type=Path,help='Optional event directory; use one directory per evaluated policy/run')
    p.add_argument('--step',type=int,help='Training iteration; defaults to checkpoint iteration')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--obstacles',type=int,help='Fixed obstacle count for pressure tests; otherwise seeded counts')
    add_scene_arguments(p)
    a=p.parse_args();torch.set_num_threads(1)
    if a.episodes<1 or a.num_envs<1:p.error('Counts must be positive')
    cfg=Config()
    if a.policy:
        meta=torch.load(a.policy,map_location='cpu',weights_only=True)['infos']
        cfg=Config.from_checkpoint(meta['avoidance_config'])
    if a.bottom:cfg.bottom=str(a.bottom.resolve())
    try:changes=apply_scene_arguments(cfg,a)
    except ValueError as exc:p.error(str(exc))
    if a.obstacles is not None and not 0<=a.obstacles<=cfg.max_obstacles:p.error('Invalid obstacle count')
    e=AvoidanceEnv(min(a.num_envs,a.episodes),a.backend,cfg)
    actor=Upper(a.policy,e.device,cfg.bottom) if a.policy else None
    start=time.monotonic();report=evaluate(e,actor,a.episodes,a.seed_base,a.zero,obstacles=a.obstacles)
    report.update(backend=a.backend,policy=str(a.policy) if a.policy else 'rule',bottom_sha256=fingerprint(cfg.bottom),
                  scene_overrides=changes,config=cfg.dict(),wall_seconds=time.monotonic()-start)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2))
    if a.tensorboard_dir:
        from torch.utils.tensorboard import SummaryWriter
        step=a.step if a.step is not None else int(torch.load(a.policy,map_location='cpu',weights_only=True)['iter']) if a.policy else 0
        with SummaryWriter(str(a.tensorboard_dir)) as writer:write_evaluation(writer,report,step)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))

if __name__=='__main__':main()
