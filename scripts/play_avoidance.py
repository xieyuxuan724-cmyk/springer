#!/usr/bin/env python3
"""Native MuJoCo playback of hierarchical avoidance or the rule baseline."""
import argparse
from pathlib import Path
from queue import SimpleQueue
import threading,time
import numpy as np
import mujoco
import torch
from beni_avoidance.common import Config,heading,rotate_xy,rays
from beni_avoidance.env import AvoidanceEnv,rule_action
from beni_avoidance.world import LayoutSamplingError
from beni_avoidance.training import Upper
from evaluate_avoidance import REASONS
from beni_avoidance.scene_cli import add_scene_arguments,apply_scene_arguments


def draw_sensors(scene,env,clear=True):
    if clear:scene.ngeom=0
    d=env.backend.datas[0]
    pos,relative,*_=env.state()
    ranges,directions=rays(pos,relative,env.centers,env.sizes,env.kinds,env.active)
    world_d=rotate_xy(directions,env.desired[:,None])[0].cpu().numpy()
    origin=d.qpos[:3].copy();origin[2]=.12
    def line(start,end,color,width=.002,kind=mujoco.mjtGeom.mjGEOM_LINE):
        if scene.ngeom>=scene.maxgeom:return
        g=scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g,kind,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,np.float32))
        mujoco.mjv_connector(g,kind,width,start,end)
        scene.ngeom+=1
    for distance,direction in zip(ranges[0].cpu().numpy(),world_d):
        end=origin.copy();end[:2]+=distance*direction
        line(origin,end,[1,.2,.1,1] if distance<.5 else [.1,.9,.65,.5],1.)
    direction=float(env.desired[0]);end=origin.copy();end[:2]+=.45*np.array([np.cos(direction),np.sin(direction)])
    line(origin,end,[1,.85,.1,1],.008,mujoco.mjtGeom.mjGEOM_ARROW)

    # Draw the same virtual limits observed by the policy; these are not walls.
    lateral=env.cfg.half_width-env.cfg.safety_radius
    angle=float(env.desired[0]);c,s=np.cos(angle),np.sin(angle)
    def world_point(x,y):return np.array([c*x-s*y,s*x+c*y,.015])
    for y in (-lateral,lateral):
        line(world_point(-env.cfg.back_margin,y),world_point(env.cfg.target_distance,y),[1,.2,.2,1],2.)
    for x,color in [(-env.cfg.back_margin,[1,.2,.2,1]),(env.cfg.target_distance,[.2,1,.2,1])]:
        line(world_point(x,-lateral),world_point(x,lateral),color,2.)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,help='Upper checkpoint; omit to show rule baseline')
    p.add_argument('--bottom',type=Path)
    p.add_argument('--seed',type=int,default=2_000_001)
    p.add_argument('--obstacles',type=int,default=4)
    p.add_argument('--speed',type=float,default=.3)
    p.add_argument('--direction-deg',type=float,default=0.)
    p.add_argument('--duration',type=float,default=0.)
    p.add_argument('--headless',action='store_true')
    p.add_argument('--screenshot',type=Path)
    add_scene_arguments(p)
    a=p.parse_args();torch.set_num_threads(1)
    if not 0<=a.speed<=.5 or not np.isfinite(a.direction_deg) or a.duration<0:p.error('Invalid speed/direction/duration')
    cfg=Config()
    if a.policy:cfg=Config.from_checkpoint(torch.load(a.policy,map_location='cpu',weights_only=True)['infos']['avoidance_config'])
    if a.bottom:cfg.bottom=str(a.bottom.resolve())
    try:apply_scene_arguments(cfg,a)
    except ValueError as exc:p.error(str(exc))
    if not 0<=a.obstacles<=cfg.max_obstacles:p.error('obstacles must be between 0 and max-obstacles')
    e=AvoidanceEnv(1,'native',cfg);actor=Upper(a.policy,'cpu',cfg.bottom) if a.policy else None
    def reset():e.reset(seeds=[a.seed],speeds=[a.speed],count=a.obstacles,directions=[np.deg2rad(a.direction_deg)])
    try:reset()
    except LayoutSamplingError as exc:p.error(str(exc))
    keys=SimpleQueue();v=None;threads=[];paused=False;finished=False
    status='running';clock=0.;max_time=a.duration or (10 if a.headless else float('inf'))
    if not a.headless:
        from mujoco import viewer
        before=set(threading.enumerate())
        v=viewer.launch_passive(e.model,e.backend.datas[0],key_callback=keys.put)
        threads=[t for t in threading.enumerate() if t not in before]
        with v.lock():
            v.cam.distance=3.5;v.cam.elevation=-30;v.cam.azimuth=135;v.opt.geomgroup[3]=0
    help_text='NUMPAD 8/2: speed | 4/6: direction+reset | 5: stop | 0: new map | .: reset | Enter: pause'
    print(help_text)
    def sync():
        if v is None:return
        start=time.monotonic()
        with v.lock():
            d=e.backend.datas[0]
            e.model.geom('floor').pos[:2]=np.floor(d.qpos[:2]/.5)*.5
            mujoco.mj_forward(e.model,d)
            v.cam.lookat[:]=e.backend.datas[0].qpos[:3]
            draw_sensors(v.user_scn,e)
        if hasattr(v,'set_texts'):
            v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                         help_text,f'{"rule" if actor is None else "policy"} | target={a.speed:.2f} m/s '
                         f'heading={a.direction_deg:.0f} deg | output={e.command[0,0]:.2f}, {e.command[0,1]:.2f} | {status}'))
        v.sync();time.sleep(max(0,.01-(time.monotonic()-start)))
    try:
        while clock<max_time and (v is None or v.is_running()):
            while not keys.empty():
                k=keys.get();restart=False
                previous_direction=a.direction_deg
                if k==328:a.speed=min(.5,a.speed+.1)
                elif k==322:a.speed=max(0.,a.speed-.1)
                elif k==325:a.speed=0.
                elif k==324:a.direction_deg+=15;restart=True
                elif k==326:a.direction_deg-=15;restart=True
                elif k==320:a.seed+=1;restart=True
                elif k==330:restart=True
                elif k==335:paused=not paused
                if restart:
                    try:reset()
                    except LayoutSamplingError as exc:
                        a.direction_deg=previous_direction
                        status='Map generation failed; old scene retained. Press numpad 0 to try next seed.'
                        print(str(exc),flush=True)
                    else:finished=False;status='running'
                else:e.speed[:]=a.speed
            if paused or finished:
                if v is None:break
                sync();continue
            action=rule_action(e) if actor is None else actor(e.observation())
            _,_,done,info=e.step(action,on_low_step=sync if v is not None else None)
            clock+=.1
            if bool(done[0]):
                status=REASONS[int(info['reason'][0])];finished=True
                print(f'Episode ended: {status}, progress={float(info["progress"][0]):.2f} m')
        if a.screenshot:
            cam=mujoco.MjvCamera();cam.lookat[:]=e.backend.datas[0].qpos[:3]
            cam.distance=3.5;cam.elevation=-45;cam.azimuth=135
            opt=mujoco.MjvOption();opt.geomgroup[3]=0
            with mujoco.Renderer(e.model,width=640,height=480) as r:
                r.update_scene(e.backend.datas[0],camera=cam,scene_option=opt)
                # Append overlays to the physical scene without discarding its geometries.
                draw_sensors(r.scene,e,clear=False)
                from PIL import Image
                a.screenshot.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(r.render()).save(a.screenshot)
    finally:
        if v:
            v.close()
            for t in threads:t.join(10)

if __name__=='__main__':main()
