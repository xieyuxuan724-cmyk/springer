import * as THREE from 'three';
import { getTrack, atDistance, crossPoint, surfaceAt } from './maps.mjs';
// Track-side structures are outside the continuous collision barriers.
// Each course has its own silhouette and architecture, independent of its palette.
export function buildScenery(parent: THREE.Group, map: any) {
  const course = getTrack(map);
  // Decoration density must not grow when small collision patches gain samples.
  const track = {
    ...course,
    frames: Array.from({ length: Math.ceil(course.length / 0.2) }, (_, i) =>
      atDistance(map, i * 0.2),
    ),
  };
  const add = (
    g: THREE.BufferGeometry,
    color: string,
    p: number[],
    roughness = 0.85,
  ) => {
    const m = new THREE.Mesh(
      g,
      new THREE.MeshStandardMaterial({ color, roughness, metalness: 0.08 }),
    );
    m.position.set(p[0], p[1], p[2]);
    m.castShadow = true;
    m.receiveShadow = true;
    parent.add(m);
    return m;
  };
  const box = (p: number[], size: number[], color: string, yaw = 0) => {
    const m = add(
      new THREE.BoxGeometry(...(size as [number, number, number])),
      color,
      p,
    );
    m.rotation.z = yaw;
    return m;
  };
  const cylinder = (
    p: number[],
    r: number,
    h: number,
    color: string,
    top = r,
    sides = 8,
  ) => {
    const m = add(new THREE.CylinderGeometry(top, r, h, sides), color, p);
    m.rotation.x = Math.PI / 2;
    return m;
  };
  const beam = (a: number[], b: number[], r: number, color: string) => {
    const av = new THREE.Vector3(...(a as [number, number, number])),
      bv = new THREE.Vector3(...(b as [number, number, number]));
    const m = add(
      new THREE.CylinderGeometry(r, r, av.distanceTo(bv), 6),
      color,
      av.clone().add(bv).multiplyScalar(0.5).toArray(),
    );
    m.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      bv.sub(av).normalize(),
    );
    return m;
  };
  const floor = (color: string, z: number) =>
    box([2, 4, z - 0.15], [110, 110, 0.3], color);
  const safe = (x: number, y: number, r: number) =>
    surfaceAt(map, x, y)!.distance > map.width / 2 + r + 0.3;
  if (map.theme === 'port') {
    floor('#315c74', -1.15);
    // Wharf foundations and quayside cargo yards.
    box([1.7, 4.8, -0.55], [5.0, 3.5, 1.15], '#667580');
    for (let i = 0; i < 8; i++) {
      const x = 0.2 + (i % 4) * 0.95,
        y = 4 + (i > 3 ? 1.05 : 0);
      if (!safe(x, y, 0.45)) continue;
      const color = ['#cd684d', '#487b91', '#e4b16d', '#596b8c'][i % 4];
      box([x, y, 0.28], [0.8, 0.7, 0.55], color);
      for (let k = 0; k < 6; k++)
        box(
          [x - 0.32 + k * 0.12, y - 0.357, 0.28],
          [0.025, 0.025, 0.5],
          '#bcd0d7',
        );
    }
    for (const [x, y] of [
      [-1, 4.6],
      [4.2, 5.3],
    ]) {
      if (!safe(x, y, 0.5)) continue;
      for (const side of [-1, 1])
        beam(
          [x, y + side * 0.4, 0],
          [x, y + side * 0.4, 2.5],
          0.055,
          '#e8be68',
        );
      beam([x - 0.6, y, 2.5], [x + 1.4, y, 2.5], 0.085, '#e8be68');
      beam([x, y, 2.7], [x + 1.4, y, 2.5], 0.025, '#e8be68');
      beam([x + 1.2, y, 2.5], [x + 1.2, y, 0.9], 0.012, '#293f50');
    }
    for (let i = 0; i < track.frames.length; i += 16) {
      const f = track.frames[i];
      cylinder(
        [f.p[0], f.p[1], (f.p[2] - 1.2) / 2],
        0.22,
        f.p[2] + 0.85,
        '#7d8992',
      );
      const pole = crossPoint(f, -f.width / 2 - 0.3);
      beam(pole, [pole[0], pole[1], pole[2] + 1.4], 0.025, '#354653');
      box([pole[0], pole[1], pole[2] + 1.4], [0.25, 0.15, 0.055], '#fff2ba');
    }
    for (let i = 0; i < 7; i++)
      box([-9 - i * 0.5, 10, -1.11], [0.15, 10, 0.01], '#4c7890');
  } else if (map.theme === 'viaduct') {
    floor('#7da9c4', -4.0);
    // Tall piers, transverse beams and underslung trusses make the climb readable.
    for (let i = 0; i < track.frames.length; i += 12) {
      const f = track.frames[i],
        next = track.frames[(i + 12) % track.frames.length];
      for (const side of [-1, 1]) {
        const p = crossPoint(f, side * (f.width / 2 - 0.15), -0.2);
        cylinder([p[0], p[1], (p[2] - 3.8) / 2], 0.13, p[2] + 3.8, '#92a8b7');
        beam(
          p,
          crossPoint(next, side * (next.width / 2 - 0.15), -0.65),
          0.045,
          '#7896aa',
        );
      }
      beam(
        crossPoint(f, -f.width / 2, -0.25),
        crossPoint(f, f.width / 2, -0.25),
        0.07,
        '#eff1dc',
      );
    }
    for (const fraction of [0.27, 0.46]) {
      const index = Math.floor(track.frames.length * fraction),
        f = track.frames[index];
      const p1 = crossPoint(f, -f.width / 2 - 0.2, 2.1),
        p2 = crossPoint(f, f.width / 2 + 0.2, 2.1);
      beam(crossPoint(f, -f.width / 2 - 0.2), p1, 0.075, '#f0e6cd');
      beam(crossPoint(f, f.width / 2 + 0.2), p2, 0.075, '#f0e6cd');
      beam(p1, p2, 0.09, '#f0e6cd');
      for (const delta of [-18, -12, -6, 6, 12, 18])
        for (const side of [-1, 1]) {
          const dest =
            track.frames[
              (index + delta + track.frames.length) % track.frames.length
            ];
          beam(
            side < 0 ? p1 : p2,
            crossPoint(dest, side * (dest.width / 2 + 0.08), 0.45),
            0.012,
            '#f2eee0',
          );
        }
    }
    for (const [x, y, z, scale] of [
      [-4, 13, -0.6, 1.8],
      [8, 18, 0.4, 2.2],
      [16, 3, -0.2, 1.9],
      [-14, 2, -0.8, 2.0],
      [4, 7, -1.8, 1.4],
      [-3, -4, -0.5, 1.6],
    ]) {
      if (!safe(x, y, scale)) continue;
      for (let i = 0; i < 5; i++) {
        const cloud = add(new THREE.SphereGeometry(1, 12, 8), '#f5fbff', [
          x + (i - 2) * scale * 0.55,
          y + (i % 2) * 0.45,
          z + (i % 3) * 0.25,
        ]);
        cloud.scale.set(scale, scale * 0.75, scale * 0.5);
        cloud.castShadow = false;
      }
    }
    for (const [x, y, z, color] of [
      [-5, 12, 4.5, '#ffadbb'],
      [14, 9, 5, '#ffd986'],
    ] as const) {
      const balloon = add(new THREE.SphereGeometry(0.75, 16, 12), color, [
        x,
        y,
        z,
      ]);
      balloon.scale.set(1, 1, 1.3);
      box([x, y, z - 1.45], [0.45, 0.45, 0.3], '#b58159');
      for (const side of [-1, 1])
        beam(
          [x + side * 0.18, y, z - 1.3],
          [x + side * 0.4, y, z - 0.6],
          0.015,
          '#fff0d9',
        );
    }
  } else if (map.theme === 'candy') {
    floor('#b5a0c8', -1.4);
    box([1, 0, -0.62], [14.4, 9.6, 0.82], '#f4d4de');
    box([1, 0, -1.07], [14.75, 9.95, 0.16], '#8c7db3');
    for (let i = 0; i < track.frames.length; i += 16) {
      const f = track.frames[i];
      if (f.p[2] < 0.35) continue;
      for (const side of [-1, 1]) {
        const p = crossPoint(f, side * (f.width / 2 + 0.2));
        // A support for the upper deck must never visually obstruct the underpass.
        if (!safe(p[0], p[1], 0.035)) continue;
        cylinder(
          [p[0], p[1], (p[2] - 0.25) / 2],
          0.065,
          p[2] + 0.25,
          '#eee7f7',
        );
      }
    }
    for (const [x, y, color] of [
      [-2.15, 0, '#83dfda'],
      [2.15, 0, '#f1a8c7'],
      [-4.5, -2.6, '#ffd584'],
      [4.5, 2.6, '#a9c5f5'],
    ] as const) {
      if (!safe(x, y, 0.35)) continue;
      cylinder([x, y, 0.52], 0.045, 1.25, '#fff3df');
      const candy = add(new THREE.TorusGeometry(0.28, 0.09, 8, 24), color, [
        x,
        y,
        1.13,
      ]);
      candy.rotation.x = Math.PI / 2;
      add(new THREE.SphereGeometry(0.12, 12, 8), '#fff2cf', [x, y, 1.13]);
    }
    for (let i = 0; i < 10; i++) {
      box(
        [-4.5 + i, -3.38, -0.15],
        [0.55, 0.16, 0.09],
        i % 2 ? '#fff0cf' : '#f28bb7',
      );
      box(
        [-4.5 + i, 3.38, -0.15],
        [0.55, 0.16, 0.09],
        i % 2 ? '#f28bb7' : '#fff0cf',
      );
    }
  }
}
