"""Native CPU and MuJoCo Warp GPU share the same compiled model."""
import numpy as np
import torch
import mujoco
from .world import model as build_world
from .common import Config


class Native:
    def __init__(self,n,cfg=None):
        cfg=cfg or Config()
        self.model=build_world(cfg);self.n=n;self.device='cpu'
        self.datas=[mujoco.MjData(self.model) for _ in range(n)]
        self.qpos=torch.zeros(n,self.model.nq);self.qvel=torch.zeros(n,self.model.nv)
        self.sensor=torch.zeros(n,self.model.nsensordata);self.ctrl=torch.zeros(n,self.model.nu)
        self.obstacle_ids=set(self.model.geom(f'obstacle_geom_{i}').id for i in range(cfg.max_obstacles))
        self.robot_ids={i for i in range(self.model.ngeom) if self.model.geom_bodyid[i]>0 and i not in self.obstacle_ids}
        self.wheel_ids={self.model.geom(f'{n}_collision').id for n in ('LW','RW')}
        self.floor=self.model.geom('floor').id
        self.collision=torch.zeros(n,dtype=torch.bool);self.ground_hit=self.collision.clone()
    def read(self):
        for i,d in enumerate(self.datas):
            self.qpos[i]=torch.from_numpy(d.qpos.copy());self.qvel[i]=torch.from_numpy(d.qvel.copy())
            self.sensor[i]=torch.from_numpy(d.sensordata.copy())
    def reset(self,ids,qpos,mocap,quat):
        for row,i in enumerate(ids.tolist()):
            d=self.datas[i];mujoco.mj_resetData(self.model,d)
            d.qpos[:]=qpos[row].cpu().numpy();d.mocap_pos[:]=mocap[row].cpu().numpy()
            d.mocap_quat[:]=quat[row].cpu().numpy()
            mujoco.mj_forward(self.model,d)
        self.read()
    def begin(self):self.collision.zero_();self.ground_hit.zero_()
    def step(self):
        for i,d in enumerate(self.datas):
            d.ctrl[:]=self.ctrl[i].numpy();mujoco.mj_step(self.model,d)
            for c in d.contact:
                if c.dist>0:continue
                a,b=int(c.geom1),int(c.geom2)
                if (a in self.obstacle_ids and b in self.robot_ids) or (b in self.obstacle_ids and a in self.robot_ids):
                    self.collision[i]=True
                robot=b if a==self.floor else a if b==self.floor else -1
                if robot in self.robot_ids and robot not in self.wheel_ids:self.ground_hit[i]=True
        self.read()
    def forward(self):
        for d in self.datas:mujoco.mj_forward(self.model,d)
        self.read()


class Warp:
    def __init__(self,n,cfg=None):
        cfg=cfg or Config()
        import warp as wp
        import mujoco_warp as mw
        from .warp_contacts import detect_contacts
        if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable; use --backend native for CPU debugging')
        self.wp,self.mw=wp,mw;self.device='cuda:0';self.n=n;self.model=build_world(cfg)
        wp.init();wp.set_device(self.device)
        self.stream=torch.cuda.Stream()
        torch.cuda.set_stream(self.stream)
        wp.set_stream(wp.stream_from_torch(self.stream))
        self.wm=mw.put_model(self.model)
        self.wd=mw.put_data(self.model,mujoco.MjData(self.model),nworld=n,nconmax=64,njmax=256)
        self.qpos=wp.to_torch(self.wd.qpos);self.qvel=wp.to_torch(self.wd.qvel)
        self.sensor=wp.to_torch(self.wd.sensordata);self.ctrl=wp.to_torch(self.wd.ctrl)
        self.mocap=wp.to_torch(self.wd.mocap_pos);self.quat=wp.to_torch(self.wd.mocap_quat)
        self.reset_mask=torch.zeros(n,device=self.device,dtype=torch.bool)
        self.reset_wp=wp.from_torch(self.reset_mask,dtype=wp.bool)
        kind=np.zeros(self.model.ngeom,np.int32)
        obstacle_ids=[self.model.geom(f'obstacle_geom_{i}').id for i in range(cfg.max_obstacles)]
        kind[self.model.geom_bodyid>0]=1;kind[obstacle_ids]=2
        for name in ('LW','RW'):kind[self.model.geom(f'{name}_collision').id]=3
        kind[self.model.geom('floor').id]=4
        self.kind=wp.array(kind,dtype=int,device=self.device)
        self.hit_wp=wp.zeros(n,dtype=int,device=self.device)
        self.ground_wp=wp.zeros(n,dtype=int,device=self.device)
        self.hit=wp.to_torch(self.hit_wp);self.ground=wp.to_torch(self.ground_wp)
        with wp.ScopedCapture() as capture:
            mw.step(self.wm,self.wd)
            wp.launch(detect_contacts,dim=self.wd.naconmax,inputs=[self.wd.nacon,self.wd.contact.geom,
                self.wd.contact.dist,self.wd.contact.worldid,self.kind,self.hit_wp,self.ground_wp])
        self.graph=capture.graph
        with wp.ScopedCapture() as capture:mw.forward(self.wm,self.wd)
        self.forward_graph=capture.graph
        with wp.ScopedCapture() as capture:mw.reset_data(self.wm,self.wd,reset=self.reset_wp)
        self.reset_graph=capture.graph
    @property
    def collision(self):return self.hit>0
    @property
    def ground_hit(self):return self.ground>0
    def reset(self,ids,qpos,mocap,quat):
        self.reset_mask.zero_();self.reset_mask[ids]=True
        self.wp.capture_launch(self.reset_graph)
        self.qpos[ids]=qpos;self.mocap[ids]=mocap;self.quat[ids]=quat
        self.forward()
    def begin(self):self.hit.zero_();self.ground.zero_()
    def step(self):self.wp.capture_launch(self.graph)
    def forward(self):self.wp.capture_launch(self.forward_graph)
    def check_capacity(self):
        if int(self.wp.to_torch(self.wd.nacon)[0])>=self.wd.naconmax:
            raise RuntimeError('Warp contact buffer exhausted; increase nconmax')
        if bool((self.wp.to_torch(self.wd.nefc)>=self.wd.njmax).any()):
            raise RuntimeError('Warp constraint buffer exhausted; increase njmax')
