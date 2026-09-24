import * as THREE from 'three';

export const EFFECT_LEVELS = {
  off: {
    label: 'Off',
    particles: 0,
    rate: 0,
    burst: 0,
    rings: 0,
    dpr: 1,
    shadow: 0,
  },
  low: {
    label: 'Low · Power saver',
    particles: 64,
    rate: 8,
    burst: 0.5,
    rings: 2,
    dpr: 1,
    shadow: 0,
  },
  medium: {
    label: 'Medium · Balanced',
    particles: 192,
    rate: 14,
    burst: 1,
    rings: 4,
    dpr: 1.5,
    shadow: 1024,
  },
  high: {
    label: 'High · Full effects',
    particles: 384,
    rate: 22,
    burst: 1.5,
    rings: 8,
    dpr: 2,
    shadow: 2048,
  },
} as const;
export type EffectsQuality = keyof typeof EFFECT_LEVELS;
export function effectsQuality(value: unknown): EffectsQuality {
  return typeof value === 'string' && Object.hasOwn(EFFECT_LEVELS, value)
    ? (value as EffectsQuality)
    : 'high';
}
export function readEffectsQuality(): EffectsQuality {
  try {
    const saved = localStorage.getItem('md_effects_quality');
    if (saved !== null) return effectsQuality(saved);
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches)
      return 'off';
  } catch {}
  return 'high';
}

type Player = {
  id: string;
  p: number[];
  q: number[];
  speed?: number;
  connected?: boolean;
  finished?: number | null;
  effects?: Record<string, number>;
  lastPickup?: {
    seq: number;
    p: number[];
    value: number;
    color: string;
    streak: number;
    boost?: boolean;
  } | null;
};
type Actor = {
  player: Player;
  finished: number | null;
  boost: number;
  trail: number;
  pickup: number;
};
type Ring = {
  mesh: THREE.Mesh<THREE.RingGeometry, THREE.MeshBasicMaterial>;
  life: number;
  duration: number;
  radius: number;
};
type Burst = 'pickup' | 'boost' | 'finish';
const GOLD = [1, 0.83, 0.28],
  CYAN = [0.26, 0.92, 1],
  PINK = [1, 0.37, 0.68];
const CONFETTI = [GOLD, CYAN, PINK, [0.55, 1, 0.58]];

// Rendering only. This class has no access to the physics worker or commands.
// One particle draw call, fixed pools, at most eight actors. No textures/postprocessing.
export class RaceEffects {
  group = new THREE.Group();
  quality: EffectsQuality = 'off';
  actors = new Map<string, Actor>();
  points?: THREE.Points<THREE.BufferGeometry, THREE.ShaderMaterial>;
  ringGeometry?: THREE.RingGeometry;
  ringMaterial?: THREE.MeshBasicMaterial;
  rings: Ring[] = [];
  labels: {
    sprite: THREE.Sprite;
    canvas: HTMLCanvasElement;
    texture: THREE.CanvasTexture;
    life: number;
  }[] = [];
  positions = new Float32Array(0);
  velocities = new Float32Array(0);
  colors = new Float32Array(0);
  sizes = new Float32Array(0);
  alphas = new Float32Array(0);
  life = new Float32Array(0);
  duration = new Float32Array(0);
  cursor = 0;
  active = 0;
  emitted = 0;
  viewportHeight = 720;
  roomKey = '';
  racing = false;
  position = new THREE.Vector3();
  constructor(quality: EffectsQuality = 'high') {
    this.group.name = 'race-effects';
    this.group.visible = false;
    this.setQuality(quality);
  }
  setQuality(value: EffectsQuality) {
    const level = effectsQuality(value);
    if (level === this.quality) return;
    this.release();
    this.quality = level;
    const budget = EFFECT_LEVELS[level];
    this.group.visible = level !== 'off';
    if (!budget.particles) return;
    const n = budget.particles;
    this.positions = new Float32Array(n * 3);
    this.velocities = new Float32Array(n * 3);
    this.colors = new Float32Array(n * 3);
    this.sizes = new Float32Array(n);
    this.alphas = new Float32Array(n);
    this.life = new Float32Array(n);
    this.duration = new Float32Array(n);
    const geometry = new THREE.BufferGeometry();
    for (const [name, array, width] of [
      ['position', this.positions, 3],
      ['color', this.colors, 3],
      ['size', this.sizes, 1],
      ['alpha', this.alphas, 1],
    ] as const)
      geometry.setAttribute(
        name,
        new THREE.BufferAttribute(array, width).setUsage(
          THREE.DynamicDrawUsage,
        ),
      );
    const material = new THREE.ShaderMaterial({
      uniforms: { viewportHeight: { value: this.viewportHeight } },
      vertexShader: `attribute float size; attribute float alpha;
        varying vec3 tint; varying float opacity; uniform float viewportHeight;
        void main() { vec4 p = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * p;
          gl_PointSize = clamp(size * viewportHeight / max(0.1, -p.z), 1.0, 40.0);
          tint = color; opacity = alpha; }`,
      fragmentShader: `varying vec3 tint; varying float opacity;
        void main() { vec2 p = abs(gl_PointCoord - 0.5) * 2.0;
          float star = 1.0 - smoothstep(0.25, 1.0, p.x + p.y);
          if (star * opacity < 0.01) discard;
          gl_FragColor = vec4(tint, star * opacity); }`,
      vertexColors: true,
      transparent: true,
      depthWrite: false,
      toneMapped: false,
    });
    this.points = new THREE.Points(geometry, material);
    this.points.frustumCulled = false;
    this.points.visible = false;
    this.group.add(this.points);
    this.ringGeometry = new THREE.RingGeometry(
      0.84,
      1,
      level === 'high' ? 48 : 24,
    );
    this.ringMaterial = new THREE.MeshBasicMaterial({
      color: '#82f4ff',
      transparent: true,
      opacity: 0.8,
      depthWrite: false,
      side: THREE.DoubleSide,
      toneMapped: false,
    });
    for (let i = 0; i < budget.rings; i++) {
      const mesh = new THREE.Mesh(this.ringGeometry, this.ringMaterial.clone());
      mesh.visible = false;
      this.group.add(mesh);
      this.rings.push({ mesh, life: 0, duration: 0.6, radius: 1 });
      if (typeof document !== 'undefined') {
        const canvas = document.createElement('canvas');
        canvas.width = 256;
        canvas.height = 96;
        const texture = new THREE.CanvasTexture(canvas);
        const sprite = new THREE.Sprite(
          new THREE.SpriteMaterial({
            map: texture,
            transparent: true,
            depthWrite: false,
            toneMapped: false,
          }),
        );
        sprite.visible = false;
        sprite.scale.set(0.32, 0.12, 1);
        this.labels.push({ sprite, canvas, texture, life: 0 });
        this.group.add(sprite);
      }
    }
  }
  resize(height: number) {
    this.viewportHeight = height;
    if (this.points)
      this.points.material.uniforms.viewportHeight.value = height;
  }
  clear() {
    this.actors.clear();
    this.life.fill(0);
    this.alphas.fill(0);
    this.active = 0;
    this.cursor = 0;
    if (this.points) this.points.visible = false;
    for (const r of this.rings) {
      r.life = 0;
      r.mesh.visible = false;
    }
    for (const label of this.labels) {
      label.life = 0;
      label.sprite.visible = false;
    }
  }
  release() {
    this.clear();
    this.points?.geometry.dispose();
    this.points?.material.dispose();
    this.ringGeometry?.dispose();
    this.ringMaterial?.dispose();
    for (const r of this.rings) r.mesh.material.dispose();
    for (const label of this.labels) {
      label.texture.dispose();
      label.sprite.material.dispose();
    }
    this.labels = [];
    this.group.clear();
    this.rings = [];
    this.points = undefined;
    this.ringGeometry = undefined;
    this.ringMaterial = undefined;
    this.positions = this.velocities = this.colors = new Float32Array(0);
    this.life = this.duration = this.sizes = this.alphas = new Float32Array(0);
  }
  syncRoom(room: any) {
    const key = room ? `${room.code}:${room.map}:${room.startAt}` : '';
    if (this.roomKey !== key || !room) {
      this.clear();
      this.roomKey = key;
    }
    this.racing = room?.phase === 'racing';
    const live = new Set<string>();
    for (const p of (room?.players ?? []).slice(0, 8) as Player[]) {
      if (p.connected === false || !p.p || !p.q) continue;
      live.add(p.id);
      let a = this.actors.get(p.id);
      if (
        a &&
        this.quality !== 'off' &&
        ['racing', 'finished'].includes(room.phase)
      ) {
        if (p.lastPickup && p.lastPickup.seq > a.pickup) {
          this.burst(p.lastPickup.p, 'pickup', p.lastPickup.color);
          this.pickupLabel(p.lastPickup);
        }
        if ((p.effects?.boost ?? 0) > Math.max(a.boost, room.now))
          this.burst(p.p, 'boost');
        if (a.finished === null && p.finished != null)
          this.burst(p.p, 'finish');
      }
      a ??= {
        player: p,
        finished: null,
        boost: 0,
        trail: 0,
        pickup: 0,
      };
      a.player = p;
      a.finished = p.finished ?? null;
      a.boost = p.effects?.boost ?? 0;
      a.pickup = p.lastPickup?.seq ?? 0;
      this.actors.set(p.id, a);
    }
    for (const id of this.actors.keys())
      if (!live.has(id)) {
        this.actors.delete(id);
      }
  }
  pickupLabel(pickup: NonNullable<Player['lastPickup']>) {
    const label = this.labels.find((l) => l.life <= 0) ?? this.labels[0];
    if (!label) return;
    const c = label.canvas.getContext('2d')!;
    c.clearRect(0, 0, 256, 96);
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.font = 'bold 48px system-ui';
    c.lineJoin = 'round';
    c.lineWidth = 9;
    c.strokeStyle = '#18304b';
    const text = `+${pickup.value}`;
    c.strokeText(text, 128, 35);
    c.fillStyle = pickup.color;
    c.fillText(text, 128, 35);
    label.texture.needsUpdate = true;
    label.life = 0.65;
    label.sprite.position.fromArray(pickup.p);
    label.sprite.position.z += 0.25;
  }
  emit(
    p: number[],
    velocity: number[],
    color: number[],
    life: number,
    size: number,
  ) {
    const n = this.life.length;
    // Saturation drops decoration instead of allocating or accumulating work.
    for (let tries = 0; tries < n; tries++) {
      const i = this.cursor;
      this.cursor = (i + 1) % n;
      if (this.life[i] > 0) continue;
      this.positions.set(p, i * 3);
      this.velocities.set(velocity, i * 3);
      this.colors.set(color, i * 3);
      this.life[i] = this.duration[i] = life;
      this.sizes[i] = size;
      this.alphas[i] = 1;
      this.emitted++;
      return;
    }
  }
  burst(p: number[], kind: Burst, tint?: string) {
    const budget = EFFECT_LEVELS[this.quality];
    if (!budget.particles) return;
    const color = tint
      ? new THREE.Color(tint).toArray()
      : kind === 'pickup'
        ? GOLD
        : CYAN;
    const count = Math.ceil(
      (kind === 'finish' ? 48 : kind === 'pickup' ? 28 : 16) * budget.burst,
    );
    for (let i = 0; i < count; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = kind === 'finish' ? 0.9 : 0.6;
      this.emit(
        [p[0], p[1], p[2] + 0.08],
        [
          Math.cos(angle) * speed,
          Math.sin(angle) * speed,
          kind === 'finish'
            ? 0.8 + Math.random() * 0.8
            : 0.1 + Math.random() * 0.4,
        ],
        kind === 'finish' ? CONFETTI[i % 4] : color,
        kind === 'finish' ? 1.5 : 0.85,
        kind === 'pickup' ? 0.1 : 0.075,
      );
    }
    const ring = this.rings.find((r) => r.life <= 0);
    if (ring) {
      ring.life = ring.duration = kind === 'pickup' ? 0.8 : 0.6;
      ring.radius = kind === 'finish' ? 0.8 : 0.65;
      ring.mesh.position.set(p[0], p[1], p[2] + 0.025);
      ring.mesh.material.color.setRGB(...(color as [number, number, number]));
    }
  }
  update(
    dt: number,
    now: number,
    locate: (id: string, target: THREE.Vector3) => boolean,
  ) {
    if (this.quality === 'off') return;
    dt = Math.max(0, Math.min(dt, 0.05));
    const budget = EFFECT_LEVELS[this.quality];
    for (const a of this.actors.values()) {
      const p = a.player;
      if (!locate(p.id, this.position)) this.position.fromArray(p.p);
      const boost = (p.effects?.boost ?? 0) > now;
      if (!this.racing || p.finished != null || (p.speed ?? 0) < 0.55) {
        a.trail = 0;
        continue;
      }
      a.trail += dt * budget.rate * (boost ? 2 : 0.65);
      if (a.trail < 1) continue;
      a.trail %= 1;
      const [w, x, y, z] = p.q;
      const yaw = Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
      const c = Math.cos(yaw),
        s = Math.sin(yaw);
      for (const side of [-1, 1])
        this.emit(
          [
            this.position.x - c * 0.08 - s * side * 0.09,
            this.position.y - s * 0.08 + c * side * 0.09,
            this.position.z - 0.055,
          ],
          [-c * 0.15, -s * 0.15, boost ? 0.14 : 0.04],
          boost ? (side === 1 ? GOLD : CYAN) : CYAN,
          boost ? 0.65 : 0.45,
          boost ? 0.09 : 0.045,
        );
    }
    for (const label of this.labels) {
      label.life = Math.max(0, label.life - dt);
      label.sprite.visible = label.life > 0;
      label.sprite.position.z += dt * 0.38;
      label.sprite.material.opacity = Math.min(1, label.life * 3);
    }
    let last = 0;
    this.active = 0;
    for (let i = 0; i < this.life.length; i++) {
      if (this.life[i] <= 0) continue;
      this.life[i] -= dt;
      this.alphas[i] = Math.max(0, this.life[i] / this.duration[i]);
      if (this.life[i] <= 0) continue;
      this.active++;
      last = i + 1;
      const k = i * 3;
      for (let axis = 0; axis < 3; axis++)
        this.positions[k + axis] += this.velocities[k + axis] * dt;
      this.velocities[k + 2] -= 0.45 * dt;
    }
    if (this.points) {
      this.points.visible = this.active > 0;
      this.points.geometry.setDrawRange(0, last);
      if (this.active)
        for (const attr of Object.values(this.points.geometry.attributes))
          attr.needsUpdate = true;
    }
    for (const r of this.rings) {
      r.life = Math.max(0, r.life - dt);
      r.mesh.visible = r.life > 0;
      if (!r.mesh.visible) continue;
      const t = 1 - r.life / r.duration;
      r.mesh.scale.setScalar(0.05 + r.radius * t);
      r.mesh.material.opacity = (1 - t) * 0.7;
    }
  }
  stats() {
    return {
      quality: this.quality,
      particles: this.active,
      budget: this.life.length,
      actors: this.actors.size,
      rings: this.rings.filter((r) => r.life > 0).length,
    };
  }
  dispose() {
    this.release();
    this.group.removeFromParent();
  }
}
