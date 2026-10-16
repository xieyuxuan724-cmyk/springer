"""Hierarchical environment shared by training, evaluation and native playback."""
import numpy as np
import torch
from .common import Config, Bottom, Motors, low_observation, heading, rotate_xy, wrap, rays, clearance
from .world import catalogue, layout
from .backend import Native, Warp
from play_native import DEFAULT


class AvoidanceEnv:
    def __init__(self,n=1,backend='native',cfg=None):
        self.cfg=cfg or Config();self.cfg.validate()
        self.backend=Native(n,self.cfg) if backend=='native' else Warp(n,self.cfg)
        self.device=self.backend.device;self.n=n;self.model=self.backend.model
        self.rng=np.random.default_rng(self.cfg.seed)
        self.bottom=Bottom(self.cfg.bottom,n,self.device)
        self.motors=Motors(self.model,n,self.device,self.cfg.delay)
        sizes,kinds=catalogue(self.cfg.max_obstacles)
        self.sizes=torch.tensor(sizes,device=self.device);self.kinds=torch.tensor(kinds,device=self.device)
        def zeros(*shape):return torch.zeros(n,*shape,device=self.device)
        self.centers=zeros(self.cfg.max_obstacles,2);self.active=zeros(self.cfg.max_obstacles).bool()
        self.desired=zeros();self.speed=zeros();self.command=zeros(2);self.previous=zeros(2)
        self.last=zeros(7);self.frames=zeros(self.cfg.stack,self.cfg.frame_dim)
        self.steps=zeros().long();self.progress=zeros();self.previous_progress=zeros()
        self.progress_ring=zeros(50);self.min_clearance=zeros();self.heading_sum=zeros()
        self.returns=zeros();self.last_seed=np.zeros(n,np.int64);self.stage=0
        self.reward_sums={}
        self.speed_error_sum=zeros();self.max_drift=zeros();self.episode_stage=zeros().long()
        self.reset(torch.arange(n,device=self.device))

    def reset(self,ids=None,seeds=None,speeds=None,count=None,directions=None):
        if ids is None:ids=torch.arange(self.n,device=self.device)
        ids=torch.as_tensor(ids,device=self.device,dtype=torch.long)
        number=len(ids)
        if not number:return self.observation()
        if seeds is None:seeds=self.rng.integers(0,1_000_000,size=number)
        if speeds is None:
            speeds=self.rng.uniform(.1,.5,number)
            if self.cfg.version==1:speeds[self.rng.random(number)<.1]=0
        if directions is None:directions=self.rng.uniform(-np.pi,np.pi,number)
        if np.any(np.asarray(speeds)<0) or np.any(np.asarray(speeds)>.5):raise ValueError('Target speed must be in [0,.5]')
        counts=[count if count is not None else (0 if self.stage==0 else int(self.rng.integers(1,3)) if self.stage==1 else int(self.rng.integers(3,self.cfg.max_obstacles+1))) for _ in range(number)]
        layouts=[layout(int(seed),c,cfg=self.cfg) for seed,c in zip(seeds,counts)]
        self.centers[ids]=torch.tensor(np.stack([x[0] for x in layouts]),device=self.device)
        self.active[ids]=torch.tensor(np.stack([x[1] for x in layouts]),device=self.device)
        self.desired[ids]=torch.as_tensor(directions,device=self.device,dtype=torch.float32)
        self.speed[ids]=torch.as_tensor(speeds,device=self.device,dtype=torch.float32)
        self.last_seed[ids.cpu().numpy()]=seeds
        q=torch.zeros(number,self.model.nq,device=self.device);q[:,2]=.14
        q[:,3]=(self.desired[ids]/2).cos();q[:,6]=(self.desired[ids]/2).sin()
        for name,value in DEFAULT.items():q[:,self.model.joint(name).qposadr[0]]=value
        mocap=torch.zeros(number,self.cfg.max_obstacles,3,device=self.device);mocap[:,:,2]=.3
        centers=self.centers[ids].clone()
        parking=torch.zeros_like(centers)
        parking[:,:,0]=self.cfg.target_distance+20.+2*torch.arange(self.cfg.max_obstacles,device=self.device)
        parking[:,:,1]=self.cfg.half_width+20.
        centers=torch.where(self.active[ids,:,None],centers,parking)
        mocap[:,:,:2]=rotate_xy(centers,self.desired[ids,None])
        quat=torch.zeros(number,self.cfg.max_obstacles,4,device=self.device)
        quat[:,:,0]=(self.desired[ids,None]/2).cos();quat[:,:,3]=(self.desired[ids,None]/2).sin()
        self.backend.reset(ids,q,mocap,quat)
        self.bottom.reset(ids);self.motors.reset(ids)
        self.last[ids]=0;self.command[ids]=0;self.previous[ids]=0
        self.steps[ids]=-5  # 0.5 s settling, not included in evaluation time.
        self.progress[ids]=0;self.previous_progress[ids]=0;self.progress_ring[ids]=0
        self.returns[ids]=0;self.heading_sum[ids]=0;self.min_clearance[ids]=10
        self.speed_error_sum[ids]=0;self.max_drift[ids]=0;self.episode_stage[ids]=self.stage
        for value in self.reward_sums.values():value[ids]=0
        self.frames[ids]=self.frame()[ids,None,:]
        return self.observation()

    def state(self):
        q,v=self.backend.qpos,self.backend.qvel
        yaw=heading(q[:,3:7]);relative=wrap(yaw-self.desired)
        pos=rotate_xy(q[:,:2],-self.desired)
        speed=v[:,0]*yaw.cos()+v[:,1]*yaw.sin()
        w,x,y,z=q[:,3:7].unbind(-1)
        world_wz=2*(x*z-w*y)*v[:,3]+2*(y*z+w*x)*v[:,4]+(1-2*(x*x+y*y))*v[:,5]
        tilt=(1-2*(x*x+y*y)).clamp(-1,1).acos()
        return pos,relative,speed,world_wz,tilt

    def frame(self):
        pos,relative,speed,wz,tilt=self.state()
        scan,_=rays(pos,relative,self.centers,self.sizes,self.kinds,self.active,self.cfg.ray_range,self.cfg.rays)
        q=self.backend.qpos
        # 36 ranges + heading sin/cos + measured speed/yaw + two gravity components + target speed + previous command.
        w,x,y,z=q[:,3:7].unbind(-1)
        extra=torch.stack((relative.sin(),relative.cos(),speed/.5,wz,
                          2*(w*y-x*z),-2*(y*z+w*x),self.speed/.5,self.command[:,0]/.5,self.command[:,1]),-1)
        if self.cfg.version==2:
            # Distances to virtual task limits, not obstacle ground-truth poses.
            lateral=self.cfg.half_width-self.cfg.safety_radius
            navigation=torch.stack(((lateral-pos[:,1])/(2*lateral),
                                    (lateral+pos[:,1])/(2*lateral),
                                    (pos[:,0]+self.cfg.back_margin)/(self.cfg.target_distance+self.cfg.back_margin),
                                    (self.cfg.target_distance-pos[:,0])/self.cfg.target_distance),-1)
            extra=torch.cat((extra,navigation),-1)
        return torch.cat((scan/self.cfg.ray_range,extra),-1)
    def observation(self):return self.frames.flatten(1).clone()

    @torch.no_grad()
    def step(self,action,on_low_step=None):
        action=torch.as_tensor(action,device=self.device,dtype=torch.float32)
        if action.shape!=(self.n,2) or not torch.isfinite(action).all():raise ValueError('Expected finite [N,2] upper action')
        action=action.clamp(-1,1)
        requested=torch.stack(((action[:,0]+1)*.5*self.speed,action[:,1]*self.cfg.yaw_limit),-1)
        requested[self.steps<0]=0
        self.backend.begin();start_pos=self.state()[0][:,0].clone();before=self.command.clone()
        for _ in range(10):
            self.command[:,0]+=(requested[:,0]-self.command[:,0]).clamp(-.03,.03)
            self.command[:,1]+=(requested[:,1]-self.command[:,1]).clamp(-.06,.06)
            obs=low_observation(self.backend.qpos,self.backend.qvel,self.backend.sensor,self.model,self.command,self.last)
            a=self.bottom(obs)
            for _ in range(5):
                self.backend.ctrl[:,self.motors.aids]=self.motors.torque(a,self.backend.qpos,self.backend.qvel)
                self.backend.step()
            self.backend.forward();self.last.copy_(a)
            if on_low_step is not None:on_low_step()
        if hasattr(self.backend,'check_capacity'):self.backend.check_capacity()
        self.steps+=1
        pos,relative,speed,wz,tilt=self.state()
        gap=clearance(pos,self.centers,self.sizes,self.kinds,self.active,self.cfg.safety_radius)
        self.progress=pos[:,0];delta=self.progress-start_pos
        warmup=self.steps<=0
        self.min_clearance=torch.minimum(self.min_clearance,gap)
        self.heading_sum+=relative.abs()*(~warmup)
        collision=self.backend.collision
        finite=torch.isfinite(self.backend.qpos).all(-1)&torch.isfinite(self.backend.qvel).all(-1)
        fall=((tilt>np.pi/3)|(self.backend.qpos[:,2]<.055)|self.backend.ground_hit|~finite)&~warmup
        bounds=((pos[:,1].abs()>self.cfg.half_width-self.cfg.safety_radius)|(pos[:,0]<-self.cfg.back_margin))&~warmup
        idx=(self.steps.clamp_min(0)%50).long();rows=torch.arange(self.n,device=self.device)
        old=self.progress_ring[rows,idx].clone();self.progress_ring[rows,idx]=self.progress
        stall=(self.steps>=50)&(self.speed>0)&((self.progress-old)<.05)
        zero=self.speed==0
        timeout=self.steps*.1 >= torch.where(zero,torch.full_like(self.speed,self.cfg.stop_timeout),self.cfg.timeout_factor*self.cfg.target_distance/self.speed.clamp_min(.01)+self.cfg.timeout_extra)
        success=((self.progress>=self.cfg.target_distance)&~zero)|(zero&timeout&(pos.norm(dim=-1)<.15)&(speed.abs()<.1))
        failure=collision|fall|bounds|stall
        success &= ~failure
        terminated=failure|success
        truncated=timeout&~terminated
        done=terminated|truncated
        # Account for each weighted contribution under the checkpoint task version.
        moving=(~zero)&(~warmup)
        terms={
            'progress':(2 if self.cfg.version==1 else .5)*delta/(self.speed*.1+.01),
            'speed_tracking':-.05*(speed-self.speed).abs()/.5,
            'heading':-.03*relative.abs(),
            'proximity':-.15*((.25-gap)/.25).clamp(0,2),
            'command_smoothness':-.05*(self.command-before).square().sum(-1),
        }
        if self.cfg.version==2:
            margin=torch.minimum(self.cfg.half_width-self.cfg.safety_radius-pos[:,1].abs(),pos[:,0]+self.cfg.back_margin)
            terms['boundary']=-self.cfg.boundary_weight*((self.cfg.boundary_margin-margin)/self.cfg.boundary_margin).clamp(0,1).square()
        terms={k:torch.where(moving,v,0.) for k,v in terms.items()}
        terms.update(stop_speed=torch.where(zero&~warmup,-.1*speed.abs(),0.),
                     stop_drift=torch.where(zero&~warmup,-.2*pos.norm(dim=-1),0.))
        for name,value in dict(success=(10 if self.cfg.version==1 else 20)*success,collision=-30*collision,fall=-20*fall,
                               bounds=(-10 if self.cfg.version==1 else -20)*bounds,stall=-5*stall,timeout=-5*truncated).items():
            terms[name]=torch.where(~warmup,value,0.).float()
        raw=sum(terms.values())
        reward=torch.nan_to_num(raw,nan=-30.,posinf=-30.,neginf=-30.)
        terms={k:torch.nan_to_num(v,nan=0.,posinf=0.,neginf=0.) for k,v in terms.items()}
        terms['numerical_correction']=reward-sum(terms.values())
        for name,value in terms.items():
            if name not in self.reward_sums:self.reward_sums[name]=torch.zeros_like(reward)
            self.reward_sums[name]+=value
        self.speed_error_sum+=(speed-self.speed).abs()*(~warmup)
        self.max_drift=torch.maximum(self.max_drift,pos.norm(dim=-1))
        self.returns+=reward
        frame=torch.nan_to_num(self.frame()).clamp(-10,10)
        self.frames=torch.cat((self.frames[:,1:],frame[:,None]),1)
        reason=torch.zeros(self.n,dtype=torch.long,device=self.device)
        for code,mask in [(1,success),(2,truncated),(3,stall),(4,bounds),(5,fall),(6,collision)]:reason[mask]=code
        info=dict(done=done,terminated=terminated,truncated=truncated,reason=reason,success=success,
                  progress=self.progress.clone(),time=self.steps.clamp_min(0)*.1,
                  clearance=self.min_clearance.clone(),heading_error=self.heading_sum/self.steps.clamp_min(1),
                  episode_return=self.returns.clone(),reward_terms=terms,
                  reward_sums={k:v.clone() for k,v in self.reward_sums.items()},
                  speed_error=self.speed_error_sum/self.steps.clamp_min(1),
                  drift=pos.norm(dim=-1),max_drift=self.max_drift.clone(),final_speed=speed.abs(),
                  events=dict(collision=collision,fall=fall,bounds=bounds,stall=stall,timeout=truncated))
        return self.observation(),reward,done,info


def rule_action(env):
    scan=env.frame()[:,:36]*env.cfg.ray_range
    heading_error=env.state()[1]
    # Front range controls braking; compare left/right sectors for free space.
    front=scan[:,14:22].min(-1).values
    left=scan[:,23:].mean(-1);right=scan[:,:13].mean(-1)
    avoid=(front<.8).float()
    side=torch.where(left>=right,1.,-1.)
    yaw=(-1.5*heading_error + avoid*side*.9).clamp(-1,1)
    fraction=((front-.22)/.6).clamp(0,1)
    return torch.stack((2*fraction-1,yaw),-1)
