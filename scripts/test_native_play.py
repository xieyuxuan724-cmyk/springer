"""Contract tests for the standalone runtime; run with unittest discover."""
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from play_native import Policy, Simulation, ACTION, DEFAULT, CONTROL_KEYS


class NativePlaybackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pt = Policy(ROOT / 'assets/models/policy.pt')

    def setUp(self):
        self.pt.reset()

    def test_no_training_imports(self):
        self.assertFalse(any(n == 'mjlab' or n.startswith('mjlab.') or n == 'beni_rl'
                             or n == 'viser' for n in sys.modules))

    def test_recurrent_onnx_equivalence_and_reset(self):
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest('Install onnxruntime for PT/ONNX comparison')
        onnx = Policy(ROOT / 'assets/models/policy.onnx')
        rng = np.random.default_rng(42)
        sequence = rng.normal(0, .1, (20, 40)).astype(np.float32)
        sequence[:, 5], sequence[:, 30], sequence[:, 32] = -1, .11, 1
        first = None
        for obs in sequence:
            actual = self.pt.predict(obs)
            if first is None:
                first = actual
            np.testing.assert_allclose(actual, onnx.predict(obs), atol=2e-4, rtol=2e-4)
        self.pt.reset()
        np.testing.assert_array_equal(self.pt.predict(sequence[0]), first)

    def test_controls_do_not_bind_native_shortcuts(self):
        # GLFW F1..F25, ASCII and arrows must not double as robot commands.
        self.assertFalse(set(CONTROL_KEYS) & set(range(0, 315)))
        self.assertEqual(set(CONTROL_KEYS.values()),
                         {'forward', 'backward', 'left', 'right', 'stop', 'jump', 'reset', 'pause'})

    def test_model_observation_and_reset(self):
        sim = Simulation(self.pt)
        np.testing.assert_array_equal(sim.model.geom('floor').size[:2], [0., 0.])
        self.assertEqual((sim.model.nq, sim.model.nv, sim.model.nu), (15, 14, 8))
        for name in DEFAULT:
            self.assertEqual(sim.model.qpos0[sim.joints[name][0]], 0.)
        sim.data.qpos[:2] = [100.13, -80.17]
        import mujoco
        mujoco.mj_forward(sim.model, sim.data)
        np.testing.assert_array_equal(sim.model.geom('floor').pos[:2], [0., 0.])
        np.testing.assert_array_equal(sim.data.geom_xpos[sim.model.geom('floor').id, :2], [0., 0.])
        sim.reset()
        np.testing.assert_array_equal(sim.model.geom('floor').pos[:2], [0., 0.])
        obs = sim.observation()
        np.testing.assert_allclose(obs[3:6], (0, 0, -1))
        np.testing.assert_allclose(obs[9:14], 0.)
        self.assertEqual(sim.request_jump(), 'not ready')
        for _ in range(10):
            sim.step()
        sim.reset()
        np.testing.assert_array_equal(sim.observation(), obs)
        np.testing.assert_array_equal(sim.policy.hidden, 0.)
        self.assertEqual(sim.data.time, 0.)

    def test_delay_torque_and_jump_phase(self):
        sim = Simulation(self.pt, delay_steps=2)
        action = np.ones(7)
        sim.apply(action)
        # Neutral hip target matches the initial pose for both delayed steps.
        self.assertEqual(sim.data.ctrl[sim.joints['L1'][2]], 0.)
        sim.apply(action)
        self.assertEqual(sim.data.ctrl[sim.joints['L1'][2]], 0.)
        sim.apply(action)
        self.assertAlmostEqual(sim.data.ctrl[sim.joints['L1'][2]], .125)
        for name in ACTION:
            cfg = sim.motors[name]
            self.assertLessEqual(abs(sim.data.ctrl[sim.joints[name][2]]),
                                 cfg.get('simulation_effort_limit', cfg['effort_limit']))
        sim.frame = 20
        np.testing.assert_allclose(sim.observation()[36:], [1, .5, .18, 0])
        sim.ready_steps = 20
        self.assertEqual(sim.request_jump(), 'accepted')
        self.assertEqual(sim.request_jump(), 'busy')


if __name__ == '__main__':
    unittest.main()
