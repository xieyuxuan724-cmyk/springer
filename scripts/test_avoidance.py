"""Geometry, frozen-policy parity and reset isolation regression checks."""
import unittest
import numpy as np
import torch
import mujoco
from beni_avoidance.common import Bottom, Motors, low_observation, rays
from beni_avoidance.world import model, layout, route_exists
from beni_avoidance.env import AvoidanceEnv
from play_native import Policy, Simulation, ROOT


class AvoidanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.env=AvoidanceEnv(2)

    def test_layout_reproducible_traversable(self):
        for seed in range(20):
            c,a=layout(seed,8)
            self.assertTrue(route_exists(c,a))
            c2,a2=layout(seed,8)
            np.testing.assert_array_equal(c,c2)
            np.testing.assert_array_equal(a,a2)

    def test_rays_agree_with_mujoco(self):
        # Use a dedicated scene so robot meshes cannot occlude sensor rays.
        m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="box" pos="1 .3 .3" size=".12 .18 .3"/><geom type="cylinder" pos=".8 -.4 .3" size=".13 .3"/></worldbody></mujoco>')
        d=mujoco.MjData(m);mujoco.mj_forward(m,d)
        scan,dirs=rays(torch.zeros(1,2),torch.zeros(1),torch.tensor([[[1.,.3],[.8,-.4]]]),torch.tensor([[.12,.18],[.13,.13]]),torch.tensor([0,1]),torch.ones(1,2,dtype=torch.bool))
        for i,direction in enumerate(dirs[0]):
            gid=np.zeros(1,np.int32)
            distance=mujoco.mj_ray(m,d,np.array([0.,0.,.12]),np.r_[direction.numpy().astype(float),0.],None,1,-1,gid)
            self.assertAlmostEqual(float(scan[0,i]),2. if distance<0 else min(2.,distance),places=5)

    def test_bottom_and_motor_parity(self):
        sim=Simulation(Policy(ROOT/'assets/models/policy.pt'))
        m=self.env.model
        q=torch.tensor(sim.data.qpos[None],dtype=torch.float32)
        v=torch.tensor(sim.data.qvel[None],dtype=torch.float32)
        sensor=torch.tensor(sim.data.sensordata[None],dtype=torch.float32)
        obs=low_observation(q,v,sensor,m,torch.zeros(1,2),torch.zeros(1,7))
        np.testing.assert_allclose(obs[0],sim.observation(),atol=1e-6)
        bottom=Bottom(ROOT/'assets/models/policy.pt',1,'cpu')
        motors=Motors(m,1,'cpu',2)
        rng=np.random.default_rng(5)
        for _ in range(8):
            obs=torch.tensor(rng.normal(0,.1,(1,40)),dtype=torch.float32)
            np.testing.assert_allclose(bottom(obs)[0],sim.policy.predict(obs[0].numpy()),atol=1e-6)
            action=rng.normal(0,.2,7).astype(np.float32)
            sim.apply(action)
            torque=motors.torque(torch.tensor(action[None]),q,v)[0]
            np.testing.assert_allclose(torque,sim.data.ctrl[motors.aids.numpy()],atol=1e-6)

    def test_partial_reset_isolation(self):
        e=self.env;e.reset(seeds=[1,2],speeds=[.3,.3],count=0,directions=[0.,0.])
        for _ in range(6):e.step(torch.zeros(2,2))
        q=e.backend.qpos[1].clone();h=e.bottom.hidden[:,1].clone()
        history=e.motors.history[:,1].clone();frames=e.frames[1].clone()
        e.reset([0],seeds=[3],speeds=[0.],count=3,directions=[1.])
        for got,want in [(e.backend.qpos[1],q),(e.bottom.hidden[:,1],h),(e.motors.history[:,1],history),(e.frames[1],frames)]:
            torch.testing.assert_close(got,want,rtol=0,atol=0)
        self.assertEqual(float(e.bottom.hidden[:,0].abs().sum()),0.)
        self.assertEqual(e.observation().shape,(2,135))

    def test_collision_latches_across_substeps(self):
        e=self.env;e.reset(seeds=[1,2],speeds=[0.,0.],count=0,directions=[0.,0.])
        b=e.backend;d=b.datas[0]
        d.mocap_pos[0]=[0.,0.,.2]
        mujoco.mj_forward(e.model,d);b.begin();b.step()
        self.assertTrue(bool(b.collision[0]))
        d.mocap_pos[0]=[10.,10.,.3];mujoco.mj_forward(e.model,d);b.step()
        self.assertTrue(bool(b.collision[0]))
        self.assertFalse(bool(b.collision[1]))
        b.begin();self.assertFalse(bool(b.collision.any()))

if __name__=='__main__':unittest.main()
