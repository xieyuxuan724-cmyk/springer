"""Scene scaling, legacy layouts and pressure-test protocol regression."""
import argparse
import unittest
from unittest.mock import patch
import numpy as np
import torch
from beni_avoidance.common import Config
from beni_avoidance.env import AvoidanceEnv,rule_action
from beni_avoidance.world import layout,route_exists,catalogue,separated
from beni_avoidance.scene_cli import add_scene_arguments,apply_scene_arguments,standard_scene
from evaluate_avoidance import evaluate


class SceneTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_scaled_layout_and_capacity(self):
        cfg=Config(target_distance=10,half_width=3,max_obstacles=16)
        c,a=layout(42,12,cfg=cfg)
        self.assertEqual(int(a.sum()),12)
        self.assertTrue(route_exists(c,a,cfg=cfg))
        self.assertTrue(np.all(c[a,0]>=cfg.obstacle_start))
        self.assertTrue(np.all(c[a,0]<=10-cfg.obstacle_end_margin))
        self.assertGreater(c[a,0].max(),3.)
        self.assertTrue(np.all(abs(c[a,1])<=3-cfg.obstacle_side_margin))
        e=AvoidanceEnv(1,cfg=cfg);e.reset(seeds=[42],speeds=[.3],count=12,directions=[0.])
        self.assertEqual(e.model.nmocap,16)
        self.assertEqual(e.active.shape,(1,16))
        self.assertEqual(e.observation().shape,(1,147))
        self.assertEqual(e.backend.datas[0].mocap_pos.shape,(16,3))
        for _ in range(6):
            obs,_,_,_=e.step(rule_action(e));self.assertTrue(bool(torch.isfinite(obs).all()))

    def test_dense_reported_scene_is_reproducible_and_traversable(self):
        cfg=Config(target_distance=3,half_width=3,max_obstacles=64)
        for seed in range(9000200,9000210):
            c,a=layout(seed,32,cfg=cfg)
            self.assertEqual(int(a.sum()),32)
            self.assertTrue(route_exists(c,a,cfg=cfg))
            c2,a2=layout(seed,32,cfg=cfg)
            np.testing.assert_array_equal(c,c2);np.testing.assert_array_equal(a,a2)

    def test_primitive_separation(self):
        sizes=np.array([[.2,.1],[.1,.1],[.2,.1]])
        kinds=np.array([0,1,0]);centers=np.zeros((3,2))
        self.assertTrue(separated(np.array([.39,0]),1,centers,[0],sizes,kinds))
        self.assertFalse(separated(np.array([.37,0]),1,centers,[0],sizes,kinds))
        self.assertTrue(separated(np.array([0,.29]),2,centers,[0],sizes,kinds))
        self.assertFalse(separated(np.array([0,.27]),2,centers,[0],sizes,kinds))
        kinds[:]=1
        self.assertTrue(separated(np.array([.39,0]),1,centers,[0],sizes,kinds))
        self.assertFalse(separated(np.array([.37,0]),1,centers,[0],sizes,kinds))

    def test_timeout_uses_distance_and_back_margin(self):
        cfg=Config(target_distance=10,half_width=3,back_margin=1.,timeout_factor=2.,timeout_extra=5.)
        e=AvoidanceEnv(1,cfg=cfg);e.reset(seeds=[1],speeds=[.5],count=0,directions=[0.])
        # Hold state fixed, isolate termination timing from bottom dynamics.
        with patch.object(e.backend,'step'),patch.object(e.backend,'forward'):
            e.steps[:]=170  # Original 3 m task would time out now.
            e.progress_ring[:]=-10
            _,_,done,info=e.step(torch.zeros(1,2));self.assertFalse(bool(info['truncated'][0]))
            e.steps[:]=449;e.progress_ring[:]=-10
            _,_,_,info=e.step(torch.zeros(1,2));self.assertTrue(bool(info['truncated'][0]))
            e.steps[:]=0;e.backend.qpos[0,0]=-.5
            _,_,_,info=e.step(torch.zeros(1,2));self.assertFalse(bool(info['events']['bounds'][0]))
            e.backend.qpos[0,0]=-1.1
            _,_,_,info=e.step(torch.zeros(1,2));self.assertTrue(bool(info['events']['bounds'][0]))

    def test_shared_cli_and_reject_invalid(self):
        p=argparse.ArgumentParser();add_scene_arguments(p)
        cfg=Config.from_checkpoint({'target_distance':3.,'half_width':1.5})
        changes=apply_scene_arguments(cfg,p.parse_args(['--target-distance','10','--field-width','6','--max-obstacles','16']))
        self.assertEqual(cfg.half_width,3.);self.assertIn('target_distance',changes)
        self.assertFalse(standard_scene(cfg));self.assertTrue(standard_scene(Config()))
        for cfg in [Config(target_distance=.5),Config(half_width=.2),Config(timeout_factor=0),Config(max_obstacles=7),Config(target_distance=float('nan'))]:
            with self.assertRaises(ValueError):cfg.validate()
        with self.assertRaises(ValueError):layout(1,9)

    def test_stress_not_standard_acceptance(self):
        cfg=Config(target_distance=10,half_width=3)
        e=AvoidanceEnv(1,cfg=cfg)
        original=e.step
        def finish(action):
            obs,reward,done,info=original(action)
            done[:]=True;info['reason'][:]=1;info['success'][:]=True
            return obs,reward,done,info
        with patch.object(e,'step',side_effect=finish):
            report=evaluate(e,None,episodes=100,seed_base=3_000_000)
        self.assertTrue(report['stress_test']);self.assertFalse(report['formal_protocol'])
        self.assertIsNone(report['acceptance_met'])

if __name__=='__main__':unittest.main()
