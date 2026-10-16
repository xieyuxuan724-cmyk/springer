"""Version compatibility and observable-boundary regression tests."""
import tempfile
import unittest
from pathlib import Path
import torch
from beni_avoidance.common import Config
from beni_avoidance.env import AvoidanceEnv
from beni_avoidance.training import Upper,runner_config
from rsl_rl.models import MLPModel
from tensordict import TensorDict


class V2Tests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_boundary_observation_and_reward(self):
        e=AvoidanceEnv(2);e.reset(seeds=[1,1],speeds=[.3,.3],count=0,directions=[0.,0.])
        e.backend.datas[1].qpos[1]=1.1;e.backend.forward()
        frame=e.frame()
        torch.testing.assert_close(frame[0,:45],frame[1,:45])
        self.assertLess(frame[1,45],frame[0,45])
        self.assertGreater(frame[1,46],frame[0,46])
        self.assertEqual(e.observation().shape,(2,147))
        e.steps[:]=0
        _,reward,_,info=e.step(torch.zeros(2,2))
        self.assertEqual(float(info['reward_terms']['boundary'][0]),0.)
        self.assertLess(float(info['reward_terms']['boundary'][1]),0.)
        torch.testing.assert_close(sum(info['reward_terms'].values()),reward)

    def test_v2_sampling_excludes_stop_but_explicit_stop_works(self):
        e=AvoidanceEnv(4)
        for _ in range(10):
            e.reset();self.assertTrue(bool((e.speed>0).all()))
        e.reset(speeds=[0.]*4,count=0)
        self.assertTrue(bool((e.speed==0).all()))
        self.assertEqual(Config.from_checkpoint({}).version,1)
        self.assertEqual(Config.from_checkpoint({}).obs_dim,135)

    def test_distribution_limit_and_checkpoint_inference_versions(self):
        for version in (1,2):
            cfg=Config(version=version)
            obs=TensorDict({'actor':torch.randn(2,cfg.obs_dim)},batch_size=[2])
            actor_cfg=runner_config(42)['actor'];actor_cfg.pop('class_name')
            actor=MLPModel(obs,{'actor':['actor']},'actor',2,**actor_cfg)
            with torch.no_grad():actor.distribution.std_param.fill_(10.)
            actor(obs,stochastic_output=True)
            self.assertLessEqual(float(actor.output_std.max()),.600001)
            actor.eval()
            meta=cfg.dict()
            if version==1:meta.pop('version')
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'actor.pt'
                torch.save(dict(infos={'avoidance_config':meta},actor_state_dict=actor.state_dict()),path)
                loaded=Upper(path)
                torch.testing.assert_close(loaded(obs['actor']),actor(obs).tanh(),rtol=0,atol=0)

if __name__=='__main__':unittest.main()
