"""Shared actor, motor model, local observations and ray geometry."""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from play_native import ACTION, OBS, DEFAULT, ROOT, Policy, motor_config


@dataclass
class Config:
    rays: int = 36
    ray_range: float = 2.
    stack: int = 3
    upper_dt: float = .1
    target_distance: float = 3.
    half_width: float = 1.5
    yaw_limit: float = 1.
    delay: int = 2
    seed: int = 42
    bottom: str = str(ROOT / 'assets/models/policy.pt')
    safety_radius: float = .19
    def validate(self):
        if self.rays != 36 or self.stack != 3 or self.upper_dt != .1:
            raise ValueError('Current deployment contract is 36 rays / 3 frames / 10 Hz')
        if not 2 <= self.delay <= 5 or not 0 < self.yaw_limit <= 1 or self.half_width <= .5:
            raise ValueError('Invalid delay, yaw limit or field width')
    @property
    def frame_dim(self): return self.rays + 9
    @property
    def obs_dim(self): return self.frame_dim * self.stack
    def dict(self): return asdict(self)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def heading(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def wrap(a): return torch.atan2(torch.sin(a), torch.cos(a))


def rotate_xy(xy, angle):
    c, s = angle.cos(), angle.sin()
    return torch.stack((c*xy[..., 0]-s*xy[..., 1], s*xy[..., 0]+c*xy[..., 1]), -1)


class Bottom(nn.Module):
    def __init__(self, path, n, device):
        super().__init__()
        loaded = Policy(Path(path))
        if loaded.onnx: raise ValueError('Batch training requires the current repository PT actor')
        self.rnn, self.mlp = loaded.rnn.to(device), loaded.mlp.to(device)
        self.register_buffer('mean', loaded.mean.to(device))
        self.register_buffer('std', loaded.std.to(device))
        self.register_buffer('hidden', torch.zeros(1, n, 512, device=device))
        self.eval().requires_grad_(False)
    @torch.no_grad()
    def forward(self, obs):
        latent, h = self.rnn(((obs-self.mean)/(self.std+.01)).unsqueeze(0), self.hidden)
        self.hidden.copy_(h)
        return self.mlp(latent[0])
    def reset(self, ids): self.hidden[:, ids] = 0


class Motors:
    def __init__(self, model, n, device, delay):
        self.names = (*ACTION, 'head_roll_joint')
        self.qids = torch.tensor([model.joint(k).qposadr[0] for k in self.names], device=device)
        self.vids = torch.tensor([model.joint(k).dofadr[0] for k in self.names], device=device)
        self.aids = torch.tensor([model.actuator(k).id for k in self.names], device=device)
        cfg = motor_config()
        def values(key, default=0):
            return torch.tensor([cfg[k].get(key, default) for k in self.names], device=device)
        self.kp, self.kd = values('stiffness'), values('damping')
        self.limit = values('effort_limit')
        self.total = torch.tensor([cfg[k].get('simulation_effort_limit', cfg[k]['effort_limit']) for k in self.names], device=device)
        self.spring = values('spring_scale')
        self.defaults = torch.tensor([DEFAULT[k] for k in self.names], device=device)
        self.scale = torch.tensor([.25,.25,.25,.25,0,0,.5,0], device=device)
        self.vscale = torch.tensor([0,0,0,0,20,20,0,0], device=device)
        self.history = torch.zeros(delay+1, n, 8, device=device)
        self.index, self.delay = 0, delay
    def reset(self, ids): self.history[:, ids] = 0
    def torque(self, action, qpos, qvel):
        self.history[self.index, :, :7] = action
        delayed = self.history[(self.index+1) % len(self.history)]
        self.index = (self.index+1) % len(self.history)
        q, v = qpos[:, self.qids], qvel[:, self.vids]
        raw = self.kp*(self.defaults+delayed*self.scale-q)+self.kd*(delayed*self.vscale-v)
        torque = raw.clamp(-self.limit, self.limit)
        a = q-1
        return (torque+self.spring*(.22*a.sin()+.15*(2*a).sin())).clamp(-self.total, self.total)


def low_observation(qpos, qvel, sensor, model, command, last):
    n = len(qpos)
    out = torch.zeros(n, 40, device=qpos.device)
    for name, start in [('imu_gyro',0), ('head_imu_gyro',33)]:
        adr = model.sensor(name).adr[0]
        out[:, start:start+3] = sensor[:, adr:adr+3]
    w,x,y,z = qpos[:,3:7].unbind(-1)
    out[:,3:6] = torch.stack((2*(w*y-x*z), -2*(y*z+w*x), -(1-2*(x*x+y*y))),-1)
    out[:,6],out[:,8] = command[:,0],command[:,1]
    for i,name in enumerate(OBS):
        if i < 5: out[:,9+i] = qpos[:,model.joint(name).qposadr[0]]-DEFAULT[name]
        out[:,14+i] = qvel[:,model.joint(name).dofadr[0]]
    out[:,21:28] = last
    out[:,30],out[:,32] = .11,1
    return out


def rays(position, yaw, centers, sizes, kinds, active, max_range=2., count=36):
    """Exact horizontal ray intersections against upright cylinders / axis-aligned boxes.

    All inputs are in the episode's target-direction frame. No obstacle poses
    enter policy observations: they are used only by this simulated sensor.
    """
    device = position.device
    angles = yaw[:,None] + torch.linspace(-np.pi/2,np.pi/2,count,device=device)[None]
    direction = torch.stack((angles.cos(),angles.sin()),-1)
    delta = centers[:,None,:,:] - position[:,None,None,:]
    dot = (delta*direction[:,:,None,:]).sum(-1)
    disc = dot.square() - delta.square().sum(-1) + sizes[None,None,:,0].square()
    near = dot-disc.clamp_min(0).sqrt()
    inside_circle = delta.square().sum(-1) <= sizes[None,None,:,0].square()
    circle = torch.where(inside_circle, 0., torch.where((disc>=0)&(near>=0), near, float('inf')))
    safe_d = direction[:,:,None,:].expand(-1,-1,centers.shape[1],-1)
    parallel = safe_d.abs()<1e-7
    denom = torch.where(parallel, torch.ones_like(safe_d), safe_d)
    lo,hi = (delta-sizes[None,None,:,:])/denom,(delta+sizes[None,None,:,:])/denom
    enter,leave = torch.minimum(lo,hi),torch.maximum(lo,hi)
    valid_parallel = delta.abs() <= sizes[None,None,:,:]
    enter = torch.where(parallel, torch.full_like(enter,-float('inf')), enter)
    leave = torch.where(parallel, torch.full_like(leave,float('inf')), leave)
    tmin,tmax = enter.max(-1).values,leave.min(-1).values
    hit = (tmax>=tmin.clamp_min(0)) & ((~parallel)|valid_parallel).all(-1)
    box = torch.where(hit,tmin.clamp_min(0),float('inf'))
    distance = torch.where(kinds[None,None,:]==0, box, circle)
    distance = torch.where(active[:,None,:],distance,float('inf')).min(-1).values
    return distance.clamp(0,max_range), direction


def clearance(position, centers, sizes, kinds, active, radius=.19):
    delta = (position[:,None,:]-centers).abs()
    box = (delta-sizes[None]).clamp_min(0).norm(dim=-1)
    circle = delta.norm(dim=-1)-sizes[None,:,0]
    distance = torch.where(kinds[None]==0,box,circle)-radius
    return torch.where(active,distance,torch.full_like(distance,10.)).min(-1).values
