#!/usr/bin/env python3
# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT
"""Standalone BENI policy playback. No imports from beni_rl, mjlab or Viser."""
from __future__ import annotations

import argparse
import ast
from collections import deque
import json
from pathlib import Path
from queue import SimpleQueue
import time
import threading
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ACTION = ('L1', 'R1', 'L2', 'R2', 'LW', 'RW', 'head_yaw_joint')
OBS = ('L1', 'R1', 'head_yaw_joint', 'L2', 'R2', 'LW', 'RW')
DEFAULT = dict(L1=.3, R1=.3, L2=.5, R2=.5, LW=0., RW=0., head_yaw_joint=0., head_roll_joint=0.)
DT, DECIMATION = .002, 5
# GLFW key codes. Letter keys are owned by MuJoCo visual/render toggles.
# Numpad codes are distinct from the top-row digits (geometry group toggles).
CONTROL_KEYS = {
    328: 'forward',   # KP8
    322: 'backward',  # KP2
    324: 'left',      # KP4
    326: 'right',     # KP6
    325: 'stop',      # KP5
    320: 'jump',      # KP0
    330: 'reset',     # KP decimal
    335: 'pause',     # KP Enter
}
CONTROL_HELP = 'NUMPAD 8/2: speed | 4/6: turn | 5: stop | 0: jump | .: reset | Enter: pause'



class Policy:
    """Current repository GRU+ELU actor, including its saved normalization.

    PT architecture is deliberately restricted to the repository's GRU-512 /
    512,256,128 ELU contract. Export other architectures to ONNX instead.
    """
    def __init__(self, path: Path):
        self.onnx = path.suffix.lower() == '.onnx'
        if self.onnx:
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            self.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
            ins, outs = self.session.get_inputs(), self.session.get_outputs()
            if (len(ins) != 2 or len(outs) != 2 or ins[0].shape != [1, 40]
                    or ins[1].shape != [1, 1, 512] or outs[0].shape != [1, 7]
                    or outs[1].shape != [1, 1, 512]
                    or any(x.type != 'tensor(float)' for x in [*ins, *outs])):
                raise ValueError('Expected BENI float32 ONNX obs[1,40], hidden[1,1,512], actions[1,7].')
            self.input_names = [x.name for x in ins]
        elif path.suffix.lower() == '.pt':
            import torch
            self.torch = torch
            torch.set_num_threads(1)
            state = torch.load(path, map_location='cpu', weights_only=True)['actor_state_dict']
            self.mean, self.std = state['obs_normalizer._mean'], state['obs_normalizer._std']
            if self.mean.shape != (1, 40) or self.std.shape != (1, 40):
                raise ValueError('Only current 40-observation BENI policies are supported.')
            self.rnn = torch.nn.GRU(40, 512, 1).eval()
            self.rnn.load_state_dict({k.removeprefix('rnn.rnn.'): v for k, v in state.items()
                                      if k.startswith('rnn.rnn.')}, strict=True)
            self.mlp = torch.nn.Sequential(torch.nn.Linear(512, 512), torch.nn.ELU(),
                torch.nn.Linear(512, 256), torch.nn.ELU(), torch.nn.Linear(256, 128),
                torch.nn.ELU(), torch.nn.Linear(128, 7)).eval()
            self.mlp.load_state_dict({k.removeprefix('mlp.'): v for k, v in state.items()
                                      if k.startswith('mlp.')}, strict=True)
        else:
            raise ValueError('Policy must be a training .pt checkpoint or exported .onnx file.')
        self.reset()

    def reset(self):
        self.hidden = np.zeros((1, 1, 512), np.float32)

    def predict(self, observation):
        obs = np.asarray(observation, dtype=np.float32).reshape(1, 40)
        if not np.isfinite(obs).all():
            raise ValueError('Nonfinite observation')
        if self.onnx:
            action, self.hidden = self.session.run(None, dict(zip(self.input_names, (obs, self.hidden))))
        else:
            t = self.torch
            with t.inference_mode():
                normalized = (t.from_numpy(obs) - self.mean) / (self.std + .01)
                latent, hidden = self.rnn(normalized.unsqueeze(0), t.from_numpy(self.hidden))
                action = self.mlp(latent.squeeze(0)).numpy()
                self.hidden = hidden.numpy()
        action = np.asarray(action).reshape(7)
        if not np.isfinite(action).all() or not np.isfinite(self.hidden).all() or np.max(np.abs(action)) > 1000:
            raise ValueError('Invalid policy action/hidden state')
        return action.copy()


def motor_config():
    """Read literal nominal motor settings without executing training imports."""
    tree = ast.parse((ROOT / 'python/beni_rl/beni_robot.py').read_text())
    motors = {}
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in ('BeniMotorCfg', 'IdealPdActuatorCfg'):
            values = {k.arg: ast.literal_eval(k.value) for k in call.keywords if k.arg}
            for name in values.pop('target_names_expr'):
                motors[name] = values.copy()
    if set(motors) != set(DEFAULT):
        raise ValueError('Robot actuator contract changed; update native player.')
    return motors


def build_model(motors, world_builder=None):
    assets = ROOT / 'assets/beni_base'
    root = ET.parse(assets / 'mjcf/beni.xml').getroot()
    root.find('compiler').attrib.update(meshdir=str(assets / 'meshes'), texturedir=str(assets / 'textures'), autolimits='true')
    ET.SubElement(root, 'option', timestep=str(DT), gravity='0 0 -9.81', integrator='implicitfast',
                  iterations='10', ls_iterations='20', ccd_iterations='50', cone='pyramidal')
    root.remove(root.find('actuator'))
    actuator = ET.SubElement(root, 'actuator')
    for joint in root.findall('.//joint'):
        name = joint.get('name')
        if name not in motors:
            continue
        cfg = motors[name]
        joint.attrib.update(ref='0', actuatorfrclimited='false', armature=str(cfg['armature']),
                            frictionloss=str(cfg['frictionloss']), damping='0')
        joint.attrib.pop('actuatorfrcrange', None)
        limit = cfg.get('simulation_effort_limit', cfg['effort_limit'])
        ET.SubElement(actuator, 'motor', name=name, joint=name, gear='1',
                      ctrllimited='true', ctrlrange=f'{-limit} {limit}')
    for name in ('LW', 'RW'):
        for geom in root.findall(f'.//body[@name="{name}_Link"]/geom'):
            if geom.get('contype', '1') != '0' or geom.get('conaffinity', '1') != '0':
                geom.attrib.update(name=f'{name}_collision', priority='1')
    asset = root.find('asset')
    ET.SubElement(asset, 'texture', name='floor_checker', type='2d', builtin='checker',
                  rgb1='.18 .24 .30', rgb2='.65 .71 .75', width='512', height='512')
    ET.SubElement(asset, 'material', name='floor_material', texture='floor_checker',
                  texuniform='true', texrepeat='2 2', reflectance='0', shininess='0')
    world = root.find('worldbody')
    # An infinite rendered plane keeps its checker texture fixed in world space.
    ET.SubElement(world, 'geom', name='floor', type='plane', size='0 0 .1',
                  friction='1 .005 .0001', material='floor_material')
    ET.SubElement(world, 'light', pos='0 -2 3', dir='0 .5 -1', directional='true')
    if world_builder is not None:
        world_builder(world)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))


class Simulation:
    def __init__(self, policy, delay_steps=2, head_roll=0., head_yaw=0.):
        self.policy, self.delay_steps = policy, delay_steps
        self.head_roll, self.head_yaw = head_roll, head_yaw
        self.motors = motor_config()
        self.model = build_model(self.motors)
        self.data = mujoco.MjData(self.model)
        self.joints = {n: (self.model.joint(n).qposadr[0], self.model.joint(n).dofadr[0],
                          self.model.actuator(n).id) for n in DEFAULT}
        self.wheels = [self.model.geom(n + '_collision').id for n in ('LW', 'RW')]
        self.weight = self.model.body_mass.sum() * 9.81
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = (0, 0, .14)
        self.data.qpos[3:7] = (1, 0, 0, 0)
        for name, (q, _, _) in self.joints.items():
            self.data.qpos[q] = DEFAULT[name]
        self.policy.reset()
        self.last_action = np.zeros(7)
        self.delay = deque(np.zeros(7) for _ in range(self.delay_steps))
        self.vx = self.wz = 0.
        self.frame = -1
        self.cooldown = self.ready_steps = self.yaw_steps = 0
        self.preparing = self.tracking = self.lifted = self.landed = False
        self.age = self.air_steps = self.stable_steps = self.prepare_steps = 0
        self.jump_vx = 0.
        self.body_contact = False
        self.max_height = .14
        mujoco.mj_forward(self.model, self.data)

    def measurement(self):
        m, d = self.model, self.data
        wheels, load, body = [False, False], 0., False
        force = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = c.geom1, c.geom2
            robot = b if m.geom_bodyid[a] == 0 else a if m.geom_bodyid[b] == 0 else -1
            if robot < 0:
                continue
            mujoco.mj_contactForce(m, d, i, force)
            if np.linalg.norm(force[:3]) <= .1:
                continue
            if robot in self.wheels:
                wheels[self.wheels.index(robot)] = True
                load += abs((c.frame.reshape(3, 3).T @ force[:3])[2])
            else:
                body = True
        w, x, y, z = d.qpos[3:7]
        tilt = np.arccos(np.clip(1 - 2*(x*x+y*y), -1, 1))
        reference = (.2848744094, .2899141610, .4957503676, .5020063519)
        error, margin = 0., float('inf')
        for name, ref in zip(ACTION[:4], reference):
            pos = d.qpos[self.joints[name][0]]
            lo, hi = m.joint(name).range
            error = max(error, abs(pos-ref))
            margin = min(margin, pos-lo, hi-pos)
        ready = (tilt < np.deg2rad(20) and abs(d.qpos[2]-.1101198718) <= .025
                 and error <= .3 and margin > .02 and all(wheels) and load >= .7*self.weight
                 and not body and not self.body_contact and np.linalg.norm(d.qvel[3:6]) < 2)
        clearance = []
        for geom in self.wheels:
            vertical = abs(d.geom_xmat[geom, 8])
            extent = .038*np.sqrt(max(0., 1-vertical**2)) + .0105*vertical
            clearance.append(d.geom_xpos[geom, 2]-extent)
        world_wz = (d.xmat[m.body('base_link').id].reshape(3, 3) @ d.qvel[3:6])[2]
        return dict(ready=bool(ready), air=not any(wheels) and not body and min(clearance) >= .01,
                    contact=any(wheels), body=body, height=float(d.qpos[2]), wz=float(world_wz))

    def request_jump(self):
        reason = 'busy' if self.tracking else 'cooldown' if self.cooldown else 'not ready' if self.ready_steps < 20 else 'accepted'
        if reason == 'accepted':
            self.tracking = self.preparing = True
            self.jump_vx = self.vx
            self.wz = 0.
            self.prepare_steps = self.age = self.air_steps = self.stable_steps = self.yaw_steps = 0
            self.lifted = self.landed = False
        return reason

    def observation(self):
        d = self.data
        obs = np.zeros(40, np.float32)
        obs[:3] = d.sensor('imu_gyro').data
        w, x, y, z = d.sensor('imu_quat').data
        obs[3:6] = (2*(w*y-x*z), -2*(y*z+w*x), -(1-2*(x*x+y*y)))
        obs[6:9] = (self.vx, 0, self.wz)
        for i, name in enumerate(OBS):
            q, v, _ = self.joints[name]
            if i < 5:
                obs[9+i] = d.qpos[q]-DEFAULT[name]
            obs[14+i] = d.qvel[v]
        obs[21:28] = self.last_action
        obs[28:31] = (0, 0, .11)
        obs[31:33] = (self.head_yaw, 1)
        obs[33:36] = d.sensor('head_imu_gyro').data
        if self.frame >= 0:
            obs[36:40] = (1, self.frame/40, .18, 0)
        return obs

    def apply(self, action):
        self.delay.append(action.copy())
        delayed = self.delay.popleft()
        for name, (q, v, act) in self.joints.items():
            cfg = self.motors[name]
            a = delayed[ACTION.index(name)] if name in ACTION else 0.
            wheel = name in ('LW', 'RW')
            target = DEFAULT[name] + a*(.5 if name == 'head_yaw_joint' else .25)
            if name == 'head_roll_joint':
                target = self.head_roll
            velocity = a*20 if wheel else 0.
            raw = cfg['stiffness']*(target-self.data.qpos[q]) + cfg['damping']*(velocity-self.data.qvel[v])
            torque = np.clip(raw, -cfg['effort_limit'], cfg['effort_limit'])
            if cfg.get('spring_scale', 0):
                angle = self.data.qpos[q]-1
                torque += cfg['spring_scale']*(.22*np.sin(angle)+.15*np.sin(2*angle))
            limit = cfg.get('simulation_effort_limit', cfg['effort_limit'])
            self.data.ctrl[act] = np.clip(torque, -limit, limit)

    def step(self, vx=0., wz=0.):
        hold = self.tracking and (self.preparing or not self.landed and self.age < 200
                                 or self.landed and self.stable_steps < 100 and self.age < 350)
        vx, wz = (self.jump_vx, 0.) if hold else (vx, wz)
        self.vx += np.clip(vx-self.vx, -.03, .03)
        self.wz += np.clip(wz-self.wz, -.06, .06)
        action = self.policy.predict(self.observation())
        self.body_contact = False
        for _ in range(DECIMATION):
            self.apply(action)
            mujoco.mj_step(self.model, self.data)
            self.body_contact |= self.measurement()['body']
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError('Nonfinite physics state')
        self.last_action = action
        s = self.measurement()
        self.max_height = max(self.max_height, s['height'])
        self.ready_steps = self.ready_steps+1 if s['ready'] else 0
        self.yaw_steps = self.yaw_steps+1 if s['ready'] and abs(s['wz']) <= .3 and abs(self.wz) < 1e-6 else 0
        self.cooldown = max(0, self.cooldown-1)
        if self.frame >= 0:
            self.frame += 1
            if self.frame >= 41:
                self.frame, self.cooldown = -1, 100
        if self.preparing:
            self.prepare_steps += 1
            if s['body'] or self.prepare_steps >= 200:
                self.preparing = self.tracking = False
                self.cooldown = 100
                print('Jump preparation cancelled: lost support or timeout')
            elif self.yaw_steps >= 20:
                self.preparing, self.frame = False, 0
        elif self.tracking:
            self.age += 1
            self.air_steps = self.air_steps+1 if s['air'] else 0
            self.lifted |= self.air_steps >= 5
            self.landed |= self.lifted and s['contact']
            self.stable_steps = self.stable_steps+1 if s['ready'] and (self.landed or self.frame < 0) else 0
            if (self.frame < 0 and self.stable_steps >= 100) or self.age >= 900:
                self.tracking = False
        return s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, default=ROOT / 'assets/models/policy.pt')
    parser.add_argument('--headless', action='store_true', help='Run without creating a window')
    parser.add_argument('--duration', type=float, default=0, help='Simulation seconds; 0 = unlimited viewer / 10 s headless')
    parser.add_argument('--vx', type=float, default=0.)
    parser.add_argument('--wz', type=float, default=0.)
    parser.add_argument('--head-yaw', type=float, default=0., help='Policy gimbal target in radians')
    parser.add_argument('--head-roll', type=float, default=0., help='Direct local roll PD target, radians')
    parser.add_argument('--delay-steps', type=int, default=2, choices=range(2, 6), help='Fixed physics-step motor delay (2..5)')
    parser.add_argument('--jump-at', type=float, default=None, help='One jump request at simulation time in seconds')
    parser.add_argument('--log', type=Path, help='Write final JSON metrics')
    args = parser.parse_args()
    values = [args.duration, args.vx, args.wz, args.head_yaw, args.head_roll]
    if not np.isfinite(values).all() or args.duration < 0 or abs(args.vx) > 2 or abs(args.wz) > 5 or abs(args.head_roll) > .05 or abs(args.head_yaw) > 5.2:
        parser.error('Invalid command: duration>=0, |vx|<=2, |wz|<=5, |head-roll|<=.05, |head-yaw|<=5.2, all finite')
    if args.jump_at is not None and (not np.isfinite(args.jump_at) or args.jump_at < 0):
        parser.error('--jump-at must be finite and nonnegative')
    policy = Policy(args.policy.expanduser())
    sim = Simulation(policy, args.delay_steps, args.head_roll, args.head_yaw)
    print(f'Native MuJoCo {mujoco.__version__}: {args.policy}; physics=500 Hz, policy=100 Hz, delay={args.delay_steps*2} ms')
    print(CONTROL_HELP)
    print('Numpad: 8/2 speed, 4/6 turn, 5 stop, 0 jump, decimal reset, Enter pause. Close with window X / Ctrl+Q.')
    duration = args.duration or (10 if args.headless else float('inf'))
    keys = SimpleQueue()
    viewer = None
    viewer_threads = []
    if not args.headless:
        from mujoco import viewer as native_viewer
        existing_threads = set(threading.enumerate())
        viewer = native_viewer.launch_passive(sim.model, sim.data, key_callback=keys.put)
        viewer_threads = [t for t in threading.enumerate() if t not in existing_threads]
        with viewer.lock():
            viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 1., -15., 135.
            viewer.opt.geomgroup[3] = 0
    paused, requested = False, False
    jump_result = None
    samples, ready_samples, lifted = 0, 0, False
    try:
        while sim.data.time < duration and (viewer is None or viewer.is_running()):
            start = time.monotonic()
            while not keys.empty():
                command = CONTROL_KEYS.get(keys.get())
                if command is None:
                    continue
                if command == 'forward': args.vx = min(2., args.vx+.1)
                elif command == 'backward': args.vx = max(-2., args.vx-.1)
                elif command == 'left': args.wz = min(5., args.wz+.2)
                elif command == 'right': args.wz = max(-5., args.wz-.2)
                elif command == 'stop': args.vx = args.wz = 0.
                elif command == 'pause': paused = not paused
                elif command == 'reset':
                    sim.reset()
                    args.vx = args.wz = 0.
                    requested = False
                    samples = ready_samples = 0
                    lifted, jump_result = False, None
                elif command == 'jump':
                    jump_result = sim.request_jump()
                    print('Jump:', jump_result)
                print(f'vx={args.vx:.2f}, wz={args.wz:.2f}, paused={paused}')
            if not paused:
                if args.jump_at is not None and not requested and sim.data.time >= args.jump_at:
                    jump_result = sim.request_jump()
                    requested = True
                    print('Scheduled jump:', jump_result)
                state = sim.step(args.vx, args.wz)
                samples += 1
                ready_samples += state['ready']
                lifted |= sim.lifted
                if samples % 100 == 0:
                    print(f't={sim.data.time:.1f}s height={state["height"]:.3f} ready={sim.ready_steps >= 20} jump={sim.tracking}')
            if viewer is not None:
                with viewer.lock():
                    viewer.cam.lookat[:] = sim.data.qpos[:3]
                if hasattr(viewer, 'set_texts'):
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,
                                      mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                      CONTROL_HELP,
                                      f'vx={args.vx:+.2f} m/s | wz={args.wz:+.2f} rad/s | '
                                      f'ready={sim.ready_steps >= 20} | paused={paused}'))
                viewer.sync()
                time.sleep(max(0., .01-(time.monotonic()-start)))
    finally:
        if viewer is not None:
            viewer.close()
            # MuJoCo close() requests exit asynchronously. Join its newly created
            # render/reload threads before Python terminates GLFW at interpreter exit.
            for thread in viewer_threads:
                thread.join(timeout=10)
                if thread.is_alive():
                    print("Warning: native viewer shutdown did not finish within 10 seconds")
    report = dict(sim_time=float(sim.data.time), position=sim.data.qpos[:3].tolist(),
                  max_base_height=sim.max_height, ready_fraction=ready_samples/max(samples, 1),
                  jump_request=jump_result, sustained_air_detected=bool(lifted),
                  finite_state=bool(np.isfinite(sim.data.qpos).all()))
    print(json.dumps(report, indent=2))
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
