import * as THREE from 'three';
const quat = (q) => new THREE.Quaternion(q[1], q[2], q[3], q[0]);
const unit = new THREE.Vector3(1, 1, 1);
// Reconstruct the authored MuJoCo body hierarchy from transmitted joint angles.
export function avatarMatrix(geom, joints) {
  const out = new THREE.Matrix4();
  for (const body of geom.chain) {
    out.multiply(
      new THREE.Matrix4().compose(
        new THREE.Vector3(...body.pos),
        quat(body.quat),
        unit,
      ),
    );
    for (const joint of body.joints) {
      const angle =
        joint.index < 0 ? 0 : (joints[joint.index] ?? joint.ref) - joint.ref;
      const p = new THREE.Vector3(...joint.pos);
      out.multiply(new THREE.Matrix4().makeTranslation(p.x, p.y, p.z));
      out.multiply(
        new THREE.Matrix4().makeRotationAxis(
          new THREE.Vector3(...joint.axis),
          angle,
        ),
      );
      out.multiply(new THREE.Matrix4().makeTranslation(-p.x, -p.y, -p.z));
    }
  }
  return out.multiply(
    new THREE.Matrix4().compose(
      new THREE.Vector3(...geom.localPos),
      quat(geom.localQuat),
      unit,
    ),
  );
}
