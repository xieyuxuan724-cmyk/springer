# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Independent standing/locomotion checks accompanying held-out recovery tests."""
from copy import deepcopy
import random

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import EventTermCfg, TerminationManager

from .jump import JumpMeasurement


@torch.inference_mode()
def evaluate_locomotion(env_cfg, actor, seed, device, trials=16):
    """Keep native termination; require a full final standing/tracking hold."""
    scenarios = dict(standing=(0.,0.), forward=(.5,0.), backward=(-.5,0.),
                     turn_left=(0.,1.), turn_right=(0.,-1.))
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    env = None
    with torch.random.fork_rng(devices=[torch.device(device)] if torch.device(device).type == 'cuda' else []):
        try:
            cfg=deepcopy(env_cfg)
            cfg.seed=seed; cfg.scene.num_envs=trials*len(scenarios)
            cfg.auto_reset=False
            cfg.terminations={}; cfg.curriculum={}; cfg.rewards={}; cfg.metrics={}
            cfg.recorders.pop('recovery_sampling', None)
            cfg.events={k:v for k,v in cfg.events.items() if k in ('reset_base','reset_joints')}
            cfg.events['reset_base'].params.update(fallen_probability=0.,pose_range={})
            for group in cfg.observations.values(): group.enable_corruption=False
            cfg.commands['jump'].play=False
            twist=cfg.commands['twist']
            twist.resampling_time_range=(1e9,1e9); twist.rel_standing_envs=0.
            twist.heading_command=False; twist.ranges.heading=None
            twist.ranges.lin_vel_x=twist.ranges.lin_vel_y=twist.ranges.ang_vel_z=(0.,0.)
            twist.pitch_range=twist.roll_range=(0.,0.); twist.height_range=(.11,.11)
            def record_termination(env,env_ids):
                terminated.logical_or_(termination.compute())
            cfg.events['record_native_termination']=EventTermCfg(func=record_termination,mode='step')
            env=ManagerBasedRlEnv(cfg,device=device)
            termination=TerminationManager(deepcopy(env_cfg.terminations),env)
            terminated=torch.zeros(env.num_envs,dtype=torch.bool,device=device)
            env.reset(seed=seed)
            target=torch.tensor(list(scenarios.values()),device=device).repeat_interleave(trials,dim=0)
            command=env.command_manager.get_term('twist')
            command.vel_command_b[:,0]=target[:,0]; command.vel_command_b[:,2]=target[:,1]
            obs=env.get_observations()
            measure=JumpMeasurement(env)
            policy=actor.as_onnx().to(device).eval()
            hidden=torch.zeros(1,env.num_envs,512,device=device)
            stable=torch.ones(env.num_envs,dtype=torch.bool,device=device)
            error=torch.zeros(env.num_envs,2,device=device)
            hold=round(1/env.step_dt); total=round(10/env.step_dt)
            for step in range(total):
                action,hidden,_=policy(obs['actor'],hidden)
                obs,*_=env.step(action)
                if step>=total-hold:
                    sample=measure(env)
                    stable &= sample['jump_ready']
                    robot=env.scene['robot'].data
                    linear=torch.stack((robot.root_link_lin_vel_b[:,0]-target[:,0],
                                        robot.root_link_lin_vel_b[:,1]),-1).norm(dim=-1)
                    yaw=(robot.root_link_ang_vel_w[:,2]-target[:,1]).abs()
                    error += torch.stack((linear,yaw),-1)/hold
            passed=stable & ~terminated & (error[:,0]<=.25) & (error[:,1]<=.3)
            rows={}
            for i,name in enumerate(scenarios):
                sl=slice(i*trials,(i+1)*trials)
                successes=int(passed[sl].sum())
                rows[name]=dict(trials=trials,successes=successes,success_rate=successes/trials,
                                passed=successes>=.9*trials,terminated=int(terminated[sl].sum()),
                                linear_error_m_s=float(error[sl,0].mean()),yaw_error_rad_s=float(error[sl,1].mean()))
            return dict(passed=all(r['passed'] for r in rows.values()),seed=seed,scenarios=rows,
                        protocol='native_termination_and_final_1s_tracking_hold',duration_s=10.)
        finally:
            if env is not None: env.close()
            random.setstate(python_rng); np.random.set_state(numpy_rng)
