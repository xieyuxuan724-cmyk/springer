import { JumpController } from './jump-controller.mjs';
import { simulationXML } from './compiled-model.mjs';
import config from './robot-config.json' with { type: 'json' };
import { spawn, getMap, surfaceAt, DEFAULT_MAP_ID } from './maps.mjs';
export const ACTION_JOINTS = [
  'L1',
  'R1',
  'L2',
  'R2',
  'LW',
  'RW',
  'head_yaw_joint',
];
export const OBS_JOINTS = [
  'L1',
  'R1',
  'head_yaw_joint',
  'L2',
  'R2',
  'LW',
  'RW',
];
export const DEFAULT_Q = {
  L1: 0.3,
  R1: 0.3,
  L2: 0.5,
  R2: 0.5,
  LW: 0,
  RW: 0,
  head_yaw_joint: 0,
  head_roll_joint: 0,
};
export const clip = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
export function springTorque(theta) {
  // Browser approximation of knee spring torque in N·m.
  const angle = theta - 1;
  return 0.22 * Math.sin(angle) + 0.15 * Math.sin(2 * angle);
}
export function motorTorque(
  name,
  position,
  velocity,
  action,
  dampingScale = 1,
) {
  const c = config.motors[name];
  const scale =
    name === 'LW' || name === 'RW'
      ? 20
      : name === 'head_yaw_joint'
        ? 0.5
        : 0.25;
  const raw =
    c.stiffness * (DEFAULT_Q[name] + action * scale - position) +
    c.damping *
      dampingScale *
      ((name === 'LW' || name === 'RW' ? action * scale : 0) - velocity);
  const limit = c.effort_limit;
  const spring = c.spring_scale ? springTorque(position) * c.spring_scale : 0;
  const totalLimit = c.simulation_effort_limit ?? c.effort_limit;
  return clip(clip(raw, -limit, limit) + spring, -totalLimit, totalLimit);
}
export class Policy {
  static async create(ort, bytes) {
    const s = await ort.InferenceSession.create(bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    try {
      const ins = s.inputMetadata,
        outs = s.outputMetadata;
      if (
        ins.length < 1 ||
        ins.length > 2 ||
        ins[0].type !== 'float32' ||
        ins[0].shape.length !== 2 ||
        ![40, 42].includes(ins[0].shape[1]) ||
        (typeof ins[0].shape[0] === 'number' && ins[0].shape[0] !== 1)
      )
        throw Error('The policy requires float32 [1, 40] or [1, 42] observations.');
      if (outs[0].type !== 'float32' || outs[0].shape.at(-1) !== 7)
        throw Error('The policy requires 7 action outputs.');
      const hidden =
        ins.length === 2
          ? ins[1].shape.map((v) => (typeof v === 'number' ? v : 1))
          : null;
      if (
        hidden &&
        (ins[1].type !== 'float32' ||
          hidden.length !== 3 ||
          hidden[1] !== 1 ||
          hidden.some((v) => v < 1 || v > 4096) ||
          outs.length < 2)
      )
        throw Error(
          'Recurrent state requires float32 [layers, 1, hidden] and matching outputs.',
        );
      const p = new Policy(ort, s, hidden, ins[0].shape[1]);
      await p.predict(new Float32Array(p.observationSize));
      p.reset();
      return p;
    } catch (e) {
      await s.release();
      throw e;
    }
  }
  constructor(ort, session, shape, observationSize = 40) {
    this.ort = ort;
    this.session = session;
    this.shape = shape;
    this.observationSize = observationSize;
    this.reset();
  }
  reset() {
    this.hidden = this.shape
      ? new Float32Array(this.shape.reduce((a, b) => a * b, 1))
      : null;
  }
  async predict(obs) {
    const feeds = {
      [this.session.inputNames[0]]: new this.ort.Tensor(
        'float32',
        obs,
        [1, this.observationSize],
      ),
    };
    if (this.hidden)
      feeds[this.session.inputNames[1]] = new this.ort.Tensor(
        'float32',
        this.hidden,
        this.shape,
      );
    let out;
    try {
      out = await this.session.run(feeds);
      const action = Float32Array.from(out[this.session.outputNames[0]].data);
      if (
        action.length !== 7 ||
        !action.every(Number.isFinite) ||
        action.some((v) => Math.abs(v) > 1000)
      )
        throw Error(
          'Invalid policy output: actions must contain 7 finite values.',
        );
      if (this.hidden) {
        const h = out[this.session.outputNames[1]];
        if (
          h.data.length !== this.hidden.length ||
          !h.data.every(Number.isFinite)
        )
          throw Error('Invalid recurrent state output shape or values.');
        this.hidden = Float32Array.from(h.data);
      }
      return action;
    } finally {
      for (const t of Object.values(feeds)) t.dispose();
      if (out) for (const t of Object.values(out)) t.dispose();
    }
  }
  async dispose() {
    await this.session.release();
  }
}
export class Simulation {
  constructor(
    mj,
    xml,
    mapId = DEFAULT_MAP_ID,
    jumpSupported = config.jumpSupported,
    jumpReference,
    compiledModel = null,
  ) {
    this.jumpSupported = jumpSupported;
    this.observationSize = 40;
    this.jumpVx = 0;
    this.mj = mj;
    this.map = getMap(mapId);
    if (compiledModel) this.model = compiledModel;
    else {
      mj.FS.writeFile('/robot/world.xml', simulationXML(xml, mapId));
      this.model = mj.MjModel.from_xml_path('/robot/world.xml');
    }
    this.data = new mj.MjData(this.model);
    this.joints = {};
    for (const name of Object.keys(DEFAULT_Q)) {
      const id = mj.mj_name2id(this.model, 3, name);
      this.joints[name] = {
        q: this.model.jnt_qposadr[id],
        v: this.model.jnt_dofadr[id],
        act: mj.mj_name2id(this.model, 19, name),
      };
    }
    this.sensors = {};
    for (const name of ['imu_gyro', 'imu_quat', 'head_imu_gyro']) {
      const id = mj.mj_name2id(this.model, 20, name);
      this.sensors[name] = this.model.sensor_adr[id];
    }
    this.jumpController = new JumpController(jumpReference);
    this.contactForce = new mj.DoubleBuffer(6);
    this.wheelGeoms = ['LW_collision', 'RW_collision'].map((n) =>
      mj.mj_name2id(this.model, 5, n),
    );
    this.weight =
      Array.from(this.model.body_mass).reduce((a, b) => a + b, 0) * 9.81;
    this.reset(spawn(this.map));
  }
  reset(at) {
    const { mj, model: m, data: d } = this;
    mj.mj_resetData(m, d);
    d.qpos.set(at.p, 0);
    d.qpos.set([Math.cos(at.yaw / 2), 0, 0, Math.sin(at.yaw / 2)], 3);
    for (const [n, j] of Object.entries(this.joints))
      d.qpos[j.q] = DEFAULT_Q[n];
    this.lastAction = new Float32Array(7);
    this.command = 0;
    this.yawCommand = 0;
    this.bodyContact = false;
    this.jumpController.reset();
    this.actionDelay = Array.from(
      { length: config.commandDelaySteps },
      () => new Float32Array(7),
    );
    mj.mj_forward(m, d);
  }
  observation(command = { vx: 0, wz: 0 }) {
    const d = this.data,
      s = d.sensordata,
      o = new Float32Array(this.observationSize ?? 40),
      qi = this.sensors.imu_quat;
    o.set(s.subarray(this.sensors.imu_gyro, this.sensors.imu_gyro + 3), 0);
    const [w, x, y, z] = s.subarray(qi, qi + 4);
    o.set(
      [2 * (w * y - x * z), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))],
      3,
    );
    o.set([command.vx ?? 0, 0, command.wz ?? 0], 6);
    OBS_JOINTS.forEach((n, i) => {
      const j = this.joints[n];
      if (i < 5) o[9 + i] = d.qpos[j.q] - DEFAULT_Q[n];
      o[14 + i] = d.qvel[j.v];
    });
    o.set(this.lastAction, 21);
    o.set([0, 0, 0.11], 28);
    o.set([0, 1], 31);
    o.set(
      s.subarray(this.sensors.head_imu_gyro, this.sensors.head_imu_gyro + 3),
      33,
    );
    if (this.jumpSupported) o.set(this.jumpController.command, 36);
    if (o.length === 42) {
      const heading = Math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z));
      o.set(this.jumpController.headingObservation(heading), 40);
    }
    return o;
  }
  get jumpPhase() {
    return this.jumpController.frame < 0 ? -1 : this.jumpController.frame / 40;
  }
  jump() {
    this.jumpController.headingEnabled = this.observationSize === 42;
    const accepted = this.jumpSupported && this.jumpController.request(
      this.observationSize === 42 ? this.jumpMeasurement() :
        { vx: this.command, wz: this.yawCommand },
    );
    if (accepted) this.jumpVx = this.command;
    return accepted;
  }
  contacts() {
    const d = this.data,
      m = this.model,
      wheels = [false, false];
    let load = 0,
      body = false;
    // The WASM getter allocates an owned contact vector; release it each sample.
    const contacts = d.contact;
    try {
      for (let i = 0; i < d.ncon; i++) {
        const c = contacts.get(i);
        try {
          const a = c.geom1,
            b = c.geom2;
          // Only robot/world contacts; ignore robot self contacts.
          const robot =
            m.geom_bodyid[a] === 0 ? b : m.geom_bodyid[b] === 0 ? a : -1;
          if (robot < 0) continue;
          this.mj.mj_contactForce(m, d, i, this.contactForce);
          const force = this.contactForce.GetView();
          if (Math.hypot(force[0], force[1], force[2]) <= 0.1) continue;
          const wheel = this.wheelGeoms.indexOf(robot);
          if (wheel < 0) body = true;
          else {
            wheels[wheel] = true;
            load += Math.abs(
              c.frame[2] * force[0] +
                c.frame[5] * force[1] +
                c.frame[8] * force[2],
            );
          }
        } finally {
          c.delete();
        }
      }
    } finally {
      contacts.delete();
    }
    return { wheels, load, body };
  }
  jumpMeasurement() {
    const d = this.data,
      m = this.model;
    const { wheels, load, body } = this.contacts();
    const floor = surfaceAt(this.map, d.qpos[0], d.qpos[1], d.qpos[2]).z;
    const height = d.qpos[2] - floor;
    const q = d.qpos,
      tilt = Math.acos(clip(1 - 2 * (q[4] * q[4] + q[5] * q[5]), -1, 1));
    const ref = {
      L1: 0.2848744094,
      R1: 0.289914161,
      L2: 0.4957503676,
      R2: 0.5020063519,
    };
    let error = 0,
      margin = Infinity;
    for (const [name, value] of Object.entries(ref)) {
      const pos = q[this.joints[name].q],
        id = this.mj.mj_name2id(m, 3, name);
      error = Math.max(error, Math.abs(pos - value));
      margin = Math.min(
        margin,
        pos - m.jnt_range[2 * id],
        m.jnt_range[2 * id + 1] - pos,
      );
    }
    const angular = Math.hypot(...d.qvel.subarray(3, 6));
    const ready =
      tilt < (20 * Math.PI) / 180 &&
      Math.abs(height - 0.1101198718) <= 0.025 &&
      error <= 0.3 &&
      margin > 0.02 &&
      wheels.every(Boolean) &&
      load >= 0.7 * this.weight &&
      !body &&
      !this.bodyContact &&
      angular < 2;
    const clearance = this.wheelGeoms.map((id) => {
      const vertical = Math.abs(d.geom_xmat[id * 9 + 8]);
      const extent =
        0.038 * Math.sqrt(Math.max(0, 1 - vertical * vertical)) +
        0.0105 * vertical;
      const p = d.geom_xpos.subarray(id * 3, id * 3 + 3);
      return p[2] - surfaceAt(this.map, p[0], p[1], p[2]).z - extent;
    });
    return {
      heading: Math.atan2(2*(q[3]*q[6]+q[4]*q[5]), 1-2*(q[5]**2+q[6]**2)),
      // MuJoCo free-joint angular qvel is body-frame; rotate its z component to world.
      wz: 2*(q[4]*q[6]-q[3]*q[5])*d.qvel[3] +
          2*(q[5]*q[6]+q[3]*q[4])*d.qvel[4] +
          (1-2*(q[4]**2+q[5]**2))*d.qvel[5],
      tilt,
      ready,
      air:
        !wheels.some(Boolean) &&
        !body &&
        !this.bodyContact &&
        clearance.every((v) => v >= 0.01),
      contact: wheels.some(Boolean),
      body,
      height,
      clearance,
    };
  }
  apply(action) {
    const d = this.data;
    this.actionDelay.push(action);
    const delayed = this.actionDelay.shift();
    d.qfrc_applied.fill(0);
    for (let i = 0; i < ACTION_JOINTS.length; i++) {
      const name = ACTION_JOINTS[i],
        j = this.joints[name];
      // Scale only knee damping to soften landings.
      const dampingScale =
        name === 'L2' || name === 'R2' ? config.kneeDampingScale : 1;
      d.ctrl[j.act] = motorTorque(
        name,
        d.qpos[j.q],
        d.qvel[j.v],
        delayed[i],
        dampingScale,
      );
    }
    const hr = this.joints.head_roll_joint;
    d.ctrl[hr.act] = motorTorque(
      'head_roll_joint',
      d.qpos[hr.q],
      d.qvel[hr.v],
      0,
    );
  }
  async step(policy, command) {
    if (this.data.qpos[2] < -3) return; // Wait for rescue instead of integrating an unbounded fall.
    command = this.jumpController.heldDrive ?? command;
    // Ramp human commands to keep the small balancing robot inside its trained envelope.
    this.command += clip((command.vx ?? 0) - this.command, -0.03, 0.03);
    this.yawCommand += clip((command.wz ?? 0) - this.yawCommand, -0.06, 0.06);
    this.observationSize = policy.observationSize;
    if (this.observationSize === 42) {
      const j = this.jumpController;
      if (j.commandHold) this.command = this.jumpVx;
      if (!j.commandHold && Math.abs(command.wz ?? 0) > 1e-6) j.headingActive = false;
      if (j.headingActive) this.yawCommand = 0;
    }
    const obs = this.observation({ vx: this.command, wz: this.yawCommand });
    const action = await policy.predict(obs);
    this.bodyContact = false;
    for (let i = 0; i < config.decimation; i++) {
      this.apply(action);
      this.mj.mj_step(this.model, this.data);
      if (this.jumpSupported) this.bodyContact ||= this.contacts().body;
    }
    this.lastAction = action;
    if (!this.data.qpos.every(Number.isFinite))
      throw Error('Invalid physics state. Use rescue or change the policy.');
    // mj_step leaves sensors/geometry one substep behind qpos/qvel. mjlab
    // refreshes them after decimation, before assembling the next observation.
    this.mj.mj_forward(this.model, this.data);
    if (this.jumpSupported) this.jumpController.update(this.jumpMeasurement());
  }
  /** @returns {Omit<import('../shared/protocol.ts').PhysicsFrame, 'running' | 'epoch' | 'wallTime'>} */
  snapshot(visualOnly = false) {
    const d = this.data;
    const ids = visualOnly
      ? (this.visualIds ??= Array.from(
          { length: this.model.ngeom },
          (_, i) => i,
        ).filter(
          (i) =>
            this.model.geom_bodyid[i] !== 0 && this.model.geom_group[i] === 1,
        ))
      : null;
    const poses = (source, stride) =>
      ids
        ? Float32Array.from(
            ids.flatMap((id) =>
              Array.from(source.subarray(id * stride, (id + 1) * stride)),
            ),
          )
        : Float32Array.from(source);
    return {
      p: Array.from(d.qpos.subarray(0, 3)),
      q: Array.from(d.qpos.subarray(3, 7)),
      j: ACTION_JOINTS.map((n) => d.qpos[this.joints[n].q]),
      speed: Math.hypot(d.qvel[0], d.qvel[1]),
      simTime: d.time,
      jump: {
        ready: this.jumpController.readySteps >= 20,
        tracking: this.jumpController.tracking,
      },
      geomIds: ids,
      geomPos: poses(d.geom_xpos, 3),
      geomMat: poses(d.geom_xmat, 9),
    };
  }
  /** @returns {import('../shared/protocol.ts').RobotGeometry[]} */
  visualModel() {
    const m = this.model,
      geoms = [];
    for (let i = 0; i < m.ngeom; i++) {
      if (m.geom_bodyid[i] === 0 || m.geom_group[i] !== 1) continue;
      const mesh = m.geom_dataid[i],
        va = m.mesh_vertadr[mesh],
        fa = m.mesh_faceadr[mesh],
        fn = m.mesh_facenum[mesh];
      const uvadr = m.mesh_texcoordadr[mesh],
        normaladr = m.mesh_normaladr[mesh];
      const vertices = [],
        uv = [],
        normals = [];
      // OBJ uses independent indices for positions, normals and texture coordinates.
      // Expand face corners so UV seams survive MuJoCo's mesh recentering.
      for (let corner = fa * 3; corner < (fa + fn) * 3; corner++) {
        const v = (va + m.mesh_face[corner]) * 3;
        vertices.push(...m.mesh_vert.subarray(v, v + 3));
        const n = (normaladr + m.mesh_facenormal[corner]) * 3;
        normals.push(...m.mesh_normal.subarray(n, n + 3));
        if (uvadr >= 0) {
          const t = (uvadr + m.mesh_facetexcoord[corner]) * 2;
          uv.push(...m.mesh_texcoord.subarray(t, t + 2));
        }
      }
      geoms.push({
        id: i,
        body: m.geom_bodyid[i],
        localPos: Array.from(m.geom_pos.subarray(i * 3, i * 3 + 3)),
        localQuat: Array.from(m.geom_quat.subarray(i * 4, i * 4 + 4)),
        chain: (() => {
          const chain = [];
          for (
            let b = m.geom_bodyid[i];
            m.body_parentid[b] > 0;
            b = m.body_parentid[b]
          ) {
            const joints = [];
            for (let k = 0; k < m.body_jntnum[b]; k++) {
              const j = m.body_jntadr[b] + k;
              joints.push({
                index: ACTION_JOINTS.findIndex(
                  (name) => this.joints[name].q === m.jnt_qposadr[j],
                ),
                axis: Array.from(m.jnt_axis.subarray(j * 3, j * 3 + 3)),
                pos: Array.from(m.jnt_pos.subarray(j * 3, j * 3 + 3)),
                ref: m.qpos0[m.jnt_qposadr[j]],
              });
            }
            chain.unshift({
              pos: Array.from(m.body_pos.subarray(b * 3, b * 3 + 3)),
              quat: Array.from(m.body_quat.subarray(b * 4, b * 4 + 4)),
              joints,
            });
          }
          return chain;
        })(),
        vertices: Float32Array.from(vertices),
        normals: Float32Array.from(normals),
        uv: Float32Array.from(uv),
        textured: uv.length > 0,
        faces: Uint32Array.from({ length: fn * 3 }, (_, j) => j),
        color: Array.from(m.geom_rgba.subarray(i * 4, i * 4 + 3)),
      });
    }
    return geoms;
  }
  dispose() {
    this.contactForce.delete();
    this.data.delete();
    this.model.delete();
  }
}
