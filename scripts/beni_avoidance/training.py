"""Adapt the existing BENI RSL-RL PPO configuration to the two-action hierarchy."""
from dataclasses import asdict
import copy
import json
from pathlib import Path
import time
import torch
from tensordict import TensorDict
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from .common import fingerprint
from .metrics import training_metrics


def runner_config(seed,steps=64):
    # Reuse this repository's optimizer, PPO clipping, GAE and logging defaults.
    from beni_rl.rl_cfg import beni_velocity_flat_ppo_runner_cfg
    cfg=asdict(beni_velocity_flat_ppo_runner_cfg())
    for name in ('actor','critic'):
        cfg[name].update(class_name='MLPModel',hidden_dims=(128,128,64),activation='elu')
        for key in ('rnn_type','rnn_hidden_dim','rnn_num_layers','cnn_cfg'):cfg[name].pop(key,None)
    cfg['actor']['distribution_cfg']['init_std']=.6
    cfg.update(seed=seed,num_steps_per_env=steps,save_interval=10,experiment_name='beni_avoidance',
               obs_groups={'actor':('actor',),'critic':('actor',)},logger='tensorboard')
    cfg['algorithm'].update(learning_rate=3e-4,gamma=.995,entropy_coef=.01)
    return cfg


class BudgetExpired(Exception):pass


class Adapter(VecEnv):
    def __init__(self,env,steps,deadline=None,log=None):
        self.env=env;self.num_envs=env.n;self.device=env.device;self.num_actions=2
        self.cfg=env.cfg.dict();self.max_episode_length=655
        self.deadline=deadline;self.steps_per_iter=steps;self.total_steps=0
        self.log=log
    @property
    def episode_length_buf(self):return self.env.steps.clamp_min(0)
    def get_observations(self):
        return TensorDict({'actor':self.env.observation()},batch_size=[self.num_envs])
    def step(self,actions):
        if self.deadline is not None and time.monotonic()>=self.deadline:raise BudgetExpired()
        iteration=self.total_steps//self.steps_per_iter
        self.env.stage=0 if iteration<5 else 1 if iteration<20 else 2
        _,reward,done,info=self.env.step(actions.tanh())
        self.total_steps+=1
        ids=done.nonzero().flatten()
        metrics=training_metrics(self.env,info,ids)
        if len(ids):
            if self.log:
                with Path(self.log).open('a') as f:
                    for i in ids.tolist():
                        f.write(json.dumps(dict(step=self.total_steps,seed=int(self.env.last_seed[i]),
                            stage=int(self.env.episode_stage[i]),speed=float(self.env.speed[i]),reason=int(info['reason'][i]),progress=float(info['progress'][i]),
                            reward=float(info['episode_return'][i]),
                            reward_terms={k:float(v[i]) for k,v in info['reward_sums'].items()},
                            drift=float(info['drift'][i]),max_drift=float(info['max_drift'][i]),
                            events={k:bool(v[i]) for k,v in info['events'].items()}))+'\n')
            self.env.reset(ids)
        # Finite-horizon navigation: exceeding the task deadline is a genuine
        # task failure. Do not bootstrap a reset observation through time_outs.
        return self.get_observations(),reward,done.long(),{'log':metrics}


class Runner(OnPolicyRunner):
    def __init__(self,env,*args,validation=None,**kwargs):
        self.validation=validation
        self.validated_iterations=set()
        self.original_config=copy.deepcopy(args[0] if args else kwargs["train_cfg"])
        super().__init__(env,*args,**kwargs)
        self.bottom_hash=fingerprint(env.env.cfg.bottom)
    def save(self,path,infos=None):
        metadata=dict(avoidance_config=self.env.env.cfg.dict(),bottom_sha256=self.bottom_hash,
                      runner_config=self.original_config,collected_upper_steps=self.env.total_steps,
                      environment_rng=self.env.env.rng.bit_generator.state,
                      torch_rng=torch.get_rng_state(),
                      cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                      resume_note='Optimizer/normalizer restored; episodes restart with cleared bottom GRU.')
        super().save(path,metadata)
        iteration=self.current_learning_iteration
        if (self.validation and self.validation['interval']>0 and iteration>0
            and iteration % self.validation['interval']==0 and iteration not in self.validated_iterations
            and Path(path).name!='model_final.pt'):
            self.validated_iterations.add(iteration)
            self.validate_checkpoint(path,iteration)

    def validate_checkpoint(self,path,iteration):
        # Separate native environment and actor: no training GRU, observation
        # normalizer, simulator or RNG state is advanced by validation.
        from .env import AvoidanceEnv
        from .metrics import write_evaluation
        from evaluate_avoidance import evaluate
        started=time.monotonic()
        if self.env.deadline is not None and started>=self.env.deadline:return
        settings=self.validation
        deadline=min(started+settings['seconds'],self.env.deadline or float('inf'))
        directory=Path(path).parent/'validation';directory.mkdir(exist_ok=True)
        try:
            with torch.random.fork_rng(devices=[]):
                cfg=copy.deepcopy(self.env.env.cfg)
                validation_env=AvoidanceEnv(min(3,settings['episodes']),'native',cfg)
                actor=Upper(path,'cpu',cfg.bottom)
                reports=[]
                for zero,episodes,seed in [(False,settings['episodes'],2_000_000),(True,settings['zero_episodes'],2_100_000)]:
                    report=evaluate(validation_env,actor,episodes,seed,zero,deadline=deadline)
                    report.update(backend='native',policy=str(path),bottom_sha256=self.bottom_hash,
                                  iteration=iteration,wall_seconds=time.monotonic()-started)
                    reports.append(report)
                # Only emit complete validation rounds, never partial success rates.
                for report in reports:
                    suffix='zero' if report['zero_command_test'] else 'navigation'
                    (directory/f'{iteration}_{suffix}.json').write_text(json.dumps(report,indent=2))
                    if self.logger.writer is not None:write_evaluation(self.logger.writer,report,iteration)
        except TimeoutError:
            (directory/f'{iteration}_incomplete.json').write_text(json.dumps(dict(
                status='budget_exhausted',iteration=iteration,seconds=time.monotonic()-started)))
            print('Validation budget exhausted; no partial success rates logged.',flush=True)


class Upper(torch.nn.Module):
    """Standalone inference for the current upper MLP checkpoint, no mjlab import."""
    def __init__(self,path,device='cpu',bottom=None):
        super().__init__()
        saved=torch.load(path,map_location='cpu',weights_only=True)
        self.metadata=saved['infos'];state=saved['actor_state_dict']
        if bottom and fingerprint(bottom)!=self.metadata['bottom_sha256']:
            raise ValueError('Bottom policy fingerprint does not match training checkpoint')
        width=state['obs_normalizer._mean'].shape[1]
        if width!=135:raise ValueError('Upper checkpoint must have 135 inputs')
        self.register_buffer('mean',state['obs_normalizer._mean'])
        self.register_buffer('std',state['obs_normalizer._std'])
        self.mlp=torch.nn.Sequential(torch.nn.Linear(width,128),torch.nn.ELU(),torch.nn.Linear(128,128),
                                    torch.nn.ELU(),torch.nn.Linear(128,64),torch.nn.ELU(),torch.nn.Linear(64,2))
        self.mlp.load_state_dict({k.removeprefix('mlp.'):v for k,v in state.items() if k.startswith('mlp.')},strict=True)
        self.to(device).eval()
    @torch.no_grad()
    def forward(self,obs):return self.mlp((obs-self.mean)/(self.std+.01)).tanh()
