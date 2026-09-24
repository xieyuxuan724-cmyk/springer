import { downloadAsset } from './asset-download';
import type { Room, RobotGeometry, PhysicsFrame } from '../shared/protocol';
import * as THREE from 'three';
import { SKINS, getSkin, skinColor, skinTextureUrl } from './skins.mjs';
import { FrameBuffer } from './frame-buffer.mjs';
import { avatarMatrix } from './avatar.mjs';
import {
  getMap,
  getTrack,
  crossPoint,
  pickups,
  heading,
  collisionTiles,
  atDistance,
  surfaceAt,
  DEFAULT_MAP_ID,
} from './maps.mjs';
import { buildScenery } from './scenery';
import {
  RaceEffects,
  EFFECT_LEVELS,
  effectsQuality,
  type EffectsQuality,
} from './race-effects';
const Z = new THREE.Vector3(0, 0, 1);
export type TextureChannel = 'map' | 'normal' | 'orm' | 'emissive';
// Everything the debug panel may switch on. Defaults match the shipped game.
export type DebugFlags = {
  freeCamera: boolean;
  captureKeys: boolean;
  wireframe: boolean;
  collision: boolean;
  grid: boolean;
  axes: boolean;
  shadows: boolean;
};
export const DEBUG_FLAGS: DebugFlags = {
  freeCamera: false,
  captureKeys: false,
  wireframe: false,
  collision: false,
  grid: false,
  axes: false,
  shadows: true,
};
export type LightTuning = {
  exposure: number;
  sun: number;
  sunAzimuth: number;
  sunElevation: number;
  ambient: number;
  fogNear: number;
  fogFar: number;
};
export const LIGHT_TUNING: LightTuning = {
  exposure: 1.3,
  sun: 3,
  // Polar form of the shipped sun position (5, −6, 12).
  sunAzimuth: -50.19,
  sunElevation: 56.9,
  ambient: 2.2,
  fogNear: 35,
  fogFar: 100,
};
export type RobotTuning = {
  roughness: number;
  metalness: number;
  emissive: number;
};
export const ROBOT_TUNING: RobotTuning = {
  // The MJCF paints the robot with specular 0.672 / shininess 0.452 and MuJoCo
  // renders the colour layer alone, so a glossy dielectric is the look to match.
  // These multiply the ORM map when that channel is switched back on.
  roughness: 0.5,
  metalness: 0,
  emissive: 0.3,
};
// Match MuJoCo's colour-only rendering by default; debug controls can enable
// the other material maps.
export const TEXTURE_CHANNELS: Record<TextureChannel, boolean> = {
  map: true,
  normal: false,
  orm: false,
  emissive: false,
};
type RobotGeom = { textured?: boolean; color: number[] };
const EXTRA_MAPS: Record<Exclude<TextureChannel, 'map'>, string> = {
  normal: '_normal',
  orm: '_orm',
  emissive: '_emissive',
};
export class Viewer {
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(50, 1, 0.02, 180);
  world = new THREE.Group();
  robot = new THREE.Group();
  remotes = new Map<string, THREE.Group>();
  showPlayerNames = false;
  robotMeshes = new Map<number, THREE.Mesh>();
  // Shared material maps load only when enabled.
  extraTextures = new Map<TextureChannel, THREE.Texture>();
  remoteTexture(path: string) {
    const texture = new THREE.Texture();
    void downloadAsset(path).then(r => r.blob()).then(async blob => {
      const url = URL.createObjectURL(blob);
      try {
        const image = await new THREE.ImageLoader().loadAsync(url);
        texture.image = image;
        texture.needsUpdate = true;
      } finally { URL.revokeObjectURL(url); }
    }).catch(error => console.error('Texture download failed', error));
    return texture;
  }
  extraTexture(channel: Exclude<TextureChannel, 'map'>) {
    let texture = this.extraTextures.get(channel);
    if (!texture) {
      texture = this.robotAtlas(
        this.remoteTexture(
          '/robot/textures/tex_beni' + EXTRA_MAPS[channel] + '.png',
        ),
        channel === 'emissive',
      );
      this.extraTextures.set(channel, texture);
    }
    return texture;
  }
  // Disable three.js's default image flip to preserve the MuJoCo UV layout.
  robotAtlas(texture: THREE.Texture, srgb: boolean) {
    texture.flipY = false;
    if (srgb) texture.colorSpace = THREE.SRGBColorSpace;
    // One still fetching its image uploads when the loader is done; flagging it
    // now would only warn about the missing image data.
    if (texture.image) texture.needsUpdate = true;
    return texture;
  }
  skinMaps = new Map<string, THREE.Texture>();
  customMap: THREE.Texture | null = null;
  // Loaded on first use, so a room only downloads the paint jobs it shows. A
  // hot-swapped colour map stands in for every skin until it is cleared.
  skinMap(id: string) {
    if (this.customMap) return this.customMap;
    const skin = getSkin(id);
    let texture = this.skinMaps.get(skin.id);
    if (!texture) {
      texture = this.robotAtlas(
        this.remoteTexture(skinTextureUrl(skin)),
        true,
      );
      this.skinMaps.set(skin.id, texture);
    }
    return texture;
  }
  robotMaterial(geom: any, skin: string) {
    const material = this.material(skinColor(getSkin(skin), geom), {
      metalness: 0.32,
      roughness: 0.45,
    });
    if (geom.textured) {
      material.userData.robotTexture = true;
      this.paintRobot(material, geom, skin);
    }
    return material;
  }
  paintRobot(m: THREE.MeshStandardMaterial, geom: RobotGeom, skin: string) {
    const c = this.channels,
      tune = this.robotTuning;
    // The atlas already carries the paint, so tinting it would double the
    // colour; skinColor only stands in when the colour channel is off.
    m.color.set(c.map ? 'white' : skinColor(getSkin(skin), geom));
    m.map = c.map ? this.skinMap(skin) : null;
    m.normalMap = c.normal ? this.extraTexture('normal') : null;
    m.emissiveMap = c.emissive ? this.extraTexture('emissive') : null;
    m.emissive.set(c.emissive ? 'white' : 'black');
    m.emissiveIntensity = c.emissive ? tune.emissive : 0;
    // The R/G/B channels are ambient occlusion, roughness and metalness.
    const orm = c.orm ? this.extraTexture('orm') : null;
    m.aoMap = orm;
    m.roughnessMap = orm;
    m.metalnessMap = orm;
    m.roughness = tune.roughness;
    m.metalness = tune.metalness;
    m.needsUpdate = true;
  }
  frame: any = null;
  frameBuffer = new FrameBuffer();
  renderPosition = new THREE.Vector3();
  renderQuaternion = new THREE.Quaternion();
  lastRender = performance.now();
  map: any;
  checkpoints: THREE.Group[] = [];
  boxes: THREE.Group[] = [];
  previewPickups: any[] = [];
  effects: RaceEffects;
  animation = 0;
  resize: ResizeObserver;
  started = performance.now();
  target = new THREE.Vector3();
  ready = false;
  room: (Room & { clockOffset: number }) | null = null;
  selfId = '';
  skin = SKINS[Math.floor(Math.random() * SKINS.length)].id;
  model: RobotGeometry[] = [];
  preview = true;
  // --- debug view state; inert until the debug panel touches it ---
  debug: DebugFlags = { ...DEBUG_FLAGS };
  lights: LightTuning = { ...LIGHT_TUNING };
  robotTuning: RobotTuning = { ...ROBOT_TUNING };
  debugCamera: { update(dt: number): void } | null = null;
  sun!: THREE.DirectionalLight;
  ambient!: THREE.HemisphereLight;
  helpers = new THREE.Group();
  collisionView = new THREE.Group();
  roadMesh: THREE.Mesh | null = null;
  channels: Record<TextureChannel, boolean> = { ...TEXTURE_CHANNELS };
  customTextures = new Set<THREE.Texture>();
  roadTexture: THREE.Texture | null = null;
  fps = 60;
  constructor(
    public element: HTMLElement,
    quality: EffectsQuality = 'high',
  ) {
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    this.effects = new RaceEffects(quality);
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.3;
    element.appendChild(this.renderer.domElement);
    this.renderer.domElement.setAttribute(
      'aria-label',
      'BENI robot 3D race course',
    );
    this.camera.up.copy(Z);
    this.scene.add(this.world, this.robot, this.effects.group);
    const sun = new THREE.DirectionalLight(0xffffff, 3);
    sun.position.set(5, -6, 12);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -28;
    sun.shadow.camera.right = 28;
    sun.shadow.camera.top = 28;
    sun.shadow.camera.bottom = -28;
    sun.shadow.normalBias = 0.008;
    this.scene.add(sun);
    this.sun = sun;
    const ambient = new THREE.HemisphereLight(0xd4eaff, 0x405054, 2.2);
    ambient.up.copy(Z);
    this.scene.add(ambient);
    this.ambient = ambient;
    this.scene.add(this.helpers, this.collisionView);
    this.helpers.visible = false;
    this.collisionView.visible = false;
    this.resize = new ResizeObserver(() => {
      const { clientWidth: w, clientHeight: h } = element;
      if (!w || !h) return;
      this.renderer.setSize(w, h);
      this.effects.resize(h * this.renderer.getPixelRatio());
      this.camera.aspect = w / h;
      this.camera.updateProjectionMatrix();
    });
    this.resize.observe(element);
    this.camera.position.set(-3, -5, 3);
    this.setEffectsQuality(quality);
    this.setMap(DEFAULT_MAP_ID);
    this.animate();
  }
  material(color: THREE.ColorRepresentation, extra: any = {}) {
    return new THREE.MeshStandardMaterial({
      color,
      roughness: 0.7,
      metalness: 0.15,
      ...extra,
    });
  }
  mesh(
    geometry: THREE.BufferGeometry,
    color: THREE.ColorRepresentation,
    p: number[],
    parent: THREE.Object3D = this.world,
  ) {
    const m = new THREE.Mesh(geometry, this.material(color));
    m.position.set(p[0], p[1], p[2]);
    m.receiveShadow = true;
    m.castShadow = true;
    parent.add(m);
    return m;
  }
  box(
    p: number[],
    s: number[],
    color: string,
    yaw = 0,
    tilt = 0,
    parent: THREE.Object3D = this.world,
  ) {
    const m = this.mesh(
      new THREE.BoxGeometry(s[0] * 2, s[1] * 2, s[2] * 2),
      color,
      p,
      parent,
    );
    m.rotation.set(0, tilt, yaw, 'ZYX');
    return m;
  }
  clear(group: THREE.Group) {
    group.traverse((o) => {
      if (o instanceof THREE.Mesh) {
        o.geometry.dispose();
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        for (const m of mats) {
          if (!m.userData.robotTexture)
            (m as THREE.MeshBasicMaterial).map?.dispose();
          m.dispose();
        }
      }
      if (o instanceof THREE.Sprite) {
        o.material.map?.dispose();
        o.material.dispose();
      }
    });
    group.clear();
  }
  label(text: string, color = '#ffffff', width = 256) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = 80;
    const c = canvas.getContext('2d')!;
    c.fillStyle = '#11181ae8';
    c.beginPath();
    c.roundRect(0, 0, width, 80, 20);
    c.fill();
    c.fillStyle = color;
    c.font = 'bold 36px system-ui';
    c.textAlign = 'center';
    c.textBaseline = 'middle';
    c.fillText(text, width / 2, 41, width - 24);
    const texture = new THREE.CanvasTexture(canvas);
    const s = new THREE.Sprite(
      new THREE.SpriteMaterial({ map: texture, depthTest: false }),
    );
    s.scale.set(0.75, 0.235, 1);
    return s;
  }
  setMap(id: string, keepView = false) {
    this.clear(this.world);
    this.effects.clear();
    this.checkpoints = [];
    this.boxes = [];
    this.map = getMap(id);
    const map = this.map;
    this.scene.background = new THREE.Color(map.sky);
    this.scene.fog = new THREE.Fog(
      map.sky,
      this.lights.fogNear,
      this.lights.fogFar,
    );
    const track = getTrack(map);
    buildScenery(this.world, map);
    const solid = (data: any, color: string, striped = false) => {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        'position',
        new THREE.Float32BufferAttribute(data.vertices, 3),
      );
      geometry.setIndex(data.faces);
      geometry.computeVertexNormals();
      const materials = striped
        ? [this.material(map.railColor), this.material(map.railAccent)]
        : this.material(color, {
            roughness: map.theme === 'glacier' ? 0.27 : 0.88,
            metalness: map.theme === 'glacier' ? 0.22 : 0.08,
          });
      if (striped) {
        const facesPerRow = (data.columns - 1) * 12 + 12;
        // One draw per stripe, not per refined terrain row.
        for (let i = 0; i < data.rows; i += 5)
          geometry.addGroup(
            i * facesPerRow,
            Math.min(5, data.rows - i) * facesPerRow,
            Math.floor(i / 5) % 2,
          );
      }
      const mesh = new THREE.Mesh(geometry, materials);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.world.add(mesh);
      return mesh;
    };
    const roadMesh = solid(track.road, map.road);
    this.roadMesh = roadMesh;
    // UVs and relief come directly from the welded road, including local crater
    // refinement and the fork. No second surface reconstruction for rendering.
    roadMesh.geometry.setAttribute(
      'uv',
      new THREE.Float32BufferAttribute(track.road.uv, 2),
    );
    const roadColor = new THREE.Color(map.road);
    const pitColor = new THREE.Color('#1d3046');
    const raisedColor = new THREE.Color('#bf883b');
    const colors = track.road.relief.flatMap((height: number) => {
      const color = roadColor
        .clone()
        .lerp(
          height < 0 ? pitColor : raisedColor,
          height < 0
            ? Math.min(0.95, Math.sqrt(-height / 0.035))
            : Math.min(0.94, height / 0.08),
        );
      return color.toArray();
    });
    roadMesh.geometry.setAttribute(
      'color',
      new THREE.Float32BufferAttribute(colors, 3),
    );
    const baseMaterial = roadMesh.material as THREE.MeshStandardMaterial;
    baseMaterial.color.set('#ffffff');
    baseMaterial.vertexColors = true;
    baseMaterial.flatShading = true;
    roadMesh.geometry.addGroup(0, track.road.faces.length, 0);
    roadMesh.material = [baseMaterial];
    // A narrow ground-coloured rim keeps tiny bowls readable from the chase
    // camera. All rings share one draw and follow the actual sampled surface.
    const rimVertices: number[] = [],
      rimFaces: number[] = [];
    for (const h of map.relief) {
      if (h.kind !== 'crater') continue;
      const centre = track.frames.findIndex((f: any) => f.u === h.u);
      const start = rimVertices.length / 3,
        visible: boolean[] = [];
      for (let i = 0; i <= 12; i++) {
        const angle = (i * Math.PI) / 6;
        for (const radius of [(h.length / 2) * 0.88, (h.length / 2) * 0.98]) {
          const f = atDistance(
            map,
            track.lengths[centre] + Math.cos(angle) * radius,
          );
          const lateral = (h.lane * f.width) / 2 + Math.sin(angle) * radius;
          visible.push(
            Math.abs(lateral) >= f.gap && Math.abs(lateral) <= f.width / 2,
          );
          rimVertices.push(...crossPoint(f, lateral, 0.002));
        }
      }
      for (let i = 0; i < 12; i++) {
        const a = i * 2;
        if (visible.slice(a, a + 4).every(Boolean))
          rimFaces.push(
            start + a,
            start + a + 2,
            start + a + 3,
            start + a,
            start + a + 3,
            start + a + 1,
          );
      }
    }
    const rimGeometry = new THREE.BufferGeometry();
    rimGeometry.setAttribute(
      'position',
      new THREE.Float32BufferAttribute(rimVertices, 3),
    );
    rimGeometry.setIndex(rimFaces);
    this.world.add(
      new THREE.Mesh(
        rimGeometry,
        new THREE.MeshBasicMaterial({
          color: map.theme === 'port' ? '#b6a189' : '#617284',
          transparent: true,
          opacity: 0.45,
          depthWrite: false,
          side: THREE.DoubleSide,
          polygonOffset: true,
          polygonOffsetFactor: -1,
          polygonOffsetUnits: -1,
        }),
      ),
    );
    for (const h of map.hazards) {
      if (!h.requiresJump) continue;
      const f = track.frames.reduce((a: any, b: any) =>
        Math.abs(a.u - h.u) < Math.abs(b.u - h.u) ? a : b,
      );
      const i = track.frames.indexOf(f);
      const launch = atDistance(map, track.lengths[i] - h.length / 2 - 0.5);
      const prompt = this.label('↑ SPACE TO JUMP', '#ffe575', 400);
      prompt.position.fromArray(crossPoint(launch, 0, 0.32));
      prompt.scale.set(0.62, 0.124, 1);
      this.world.add(prompt);
    }
    for (const rail of track.rails) solid(rail, map.railColor, true);
    for (const structure of track.structures)
      solid(structure.mesh, structure.color);
    for (const rail of track.innerRails) solid(rail, map.railColor, true);
    for (const island of track.islands) solid(island, map.railAccent);
    for (const split of map.splits ?? []) {
      const f = track.frames.reduce((a: any, b: any) =>
        Math.abs(a.u - (split.u - split.halfU)) <
        Math.abs(b.u - (split.u - split.halfU))
          ? a
          : b,
      );
      const sign = this.label('← GARDEN  /  TUNNEL →', '#fff0bd', 480);
      sign.position.fromArray(
        crossPoint(f, 0, (split.height ?? map.railHeight) + 0.19),
      );
      sign.scale.set(0.82, 0.137, 1);
      this.world.add(sign);
    }
    // Edge rumble strips and a broken centre line follow exactly the same sampled surface.
    const ribbon = (lo: number, hi: number, color: string, broken = false) => {
      const vertices: number[] = [],
        indices: number[] = [];
      for (let i = 0; i < track.frames.length; i++) {
        const f = track.frames[i];
        vertices.push(
          ...crossPoint(f, (lo * f.width) / 2, 0.006),
          ...crossPoint(f, (hi * f.width) / 2, 0.006),
        );
        const j = (i + 1) % track.frames.length;
        if (
          !broken ||
          (!f.gap && !track.frames[j].gap && Math.floor(i / 3) % 2 === 0)
        )
          indices.push(i * 2, j * 2, j * 2 + 1, i * 2, j * 2 + 1, i * 2 + 1);
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
      g.setIndex(indices);
      g.computeVertexNormals();
      const m = new THREE.Mesh(g, this.material(color));
      this.world.add(m);
    };
    ribbon(-0.96, -0.86, map.color);
    ribbon(0.86, 0.96, map.color);
    if (map.theme !== 'glacier') ribbon(-0.012, 0.012, '#e8e3cd', true);
    // Functional arches identify the ordered checkpoints.
    map.points.forEach((p: number[], i: number) => {
      const g = new THREE.Group();
      g.position.set(p[0], p[1], p[2]);
      g.rotation.z = heading(map, i);
      const width = atDistance(map, track.checkpointLengths[i]).width;
      this.world.add(g);
      for (const side of [-1, 1])
        this.box(
          [0, side * (width / 2 + 0.06), 0.43],
          [0.025, 0.025, 0.43],
          i === 0 ? '#ffffff' : map.color,
          0,
          0,
          g,
        );
      this.box(
        [0, 0, 0.86],
        [0.03, width / 2 + 0.085, 0.025],
        i === 0 ? '#ffffff' : map.color,
        0,
        0,
        g,
      );
      this.checkpoints.push(g);
    });
    // Start line is transformed onto its own road frame, including elevation.
    const start = atDistance(map, track.checkpointLengths[0]);
    const startGroup = new THREE.Group();
    startGroup.position.fromArray(start.p);
    startGroup.rotation.z = start.yaw;
    this.world.add(startGroup);
    for (let x = 0; x < 3; x++)
      for (let y = 0; y < 16; y++)
        this.box(
          [
            -0.06 + x * 0.05,
            -start.width / 2 + ((y + 0.5) * start.width) / 16,
            0.008,
          ],
          [0.025, start.width / 32, 0.004],
          (x + y) % 2 ? '#20282b' : '#ffffff',
          0,
          0,
          startGroup,
        );
    this.previewPickups = pickups(map);
    for (let i = 0; i < 30; i++) {
      const group = new THREE.Group();
      const gem = new THREE.Mesh(
        new THREE.OctahedronGeometry(0.12),
        new THREE.MeshStandardMaterial({
          color: '#fff3c2',
          emissive: '#ffe66d',
          emissiveIntensity: 1.2,
          roughness: 0.22,
          metalness: 0.2,
        }),
      );
      const halo = new THREE.Mesh(
        new THREE.RingGeometry(0.15, 0.21, 24),
        new THREE.MeshBasicMaterial({
          color: '#ffe66d',
          transparent: true,
          opacity: 0.5,
          depthWrite: false,
          side: THREE.DoubleSide,
          toneMapped: false,
        }),
      );
      halo.position.z = -0.11;
      group.add(gem, halo);
      group.visible = false;
      this.world.add(group);
      this.boxes.push(group);
    }
    if (!keepView) {
      this.target.set(1.5, 2, 0.1);
      this.preview = true;
    }
    this.applyDebug();
  }
  // ---------------------------------------------------------------- debug view
  // Re-applies every debug toggle to freshly built geometry.
  applyDebug() {
    const d = this.debug;
    this.renderer.shadowMap.enabled =
      d.shadows && EFFECT_LEVELS[this.effects.quality].shadow > 0;
    this.setWireframe(d.wireframe);
    this.helpers.visible = d.grid || d.axes;
    if (this.helpers.visible) this.buildHelpers();
    this.collisionView.visible = d.collision;
    if (d.collision) this.buildCollisionView();
    this.applyLights();
    if (this.roadTexture) this.setRoadTexture(this.roadTexture);
    this.applyDebugMaterials();
  }
  // Update wireframe and materials without rebuilding the scene.
  applyDebugMaterials() {
    this.setWireframe(this.debug.wireframe);
    this.repaintRobots();
  }
  setFlag<K extends keyof DebugFlags>(key: K, value: DebugFlags[K]) {
    this.debug[key] = value;
    this.applyDebug();
  }
  setWireframe(on: boolean) {
    for (const root of [this.world, this.robot, ...this.remotes.values()])
      root.traverse((o) => {
        if (!(o instanceof THREE.Mesh)) return;
        for (const m of Array.isArray(o.material) ? o.material : [o.material])
          if ('wireframe' in m)
            (m as THREE.MeshStandardMaterial).wireframe = on;
      });
  }
  applyLights() {
    const l = this.lights;
    this.renderer.toneMappingExposure = l.exposure;
    this.sun.intensity = l.sun;
    this.ambient.intensity = l.ambient;
    const az = (l.sunAzimuth * Math.PI) / 180,
      el = (l.sunElevation * Math.PI) / 180,
      r = 14.318;
    this.sun.position.set(
      r * Math.cos(el) * Math.cos(az),
      r * Math.cos(el) * Math.sin(az),
      r * Math.sin(el),
    );
    if (this.scene.fog instanceof THREE.Fog) {
      this.scene.fog.near = l.fogNear;
      this.scene.fog.far = Math.max(l.fogNear + 1, l.fogFar);
    }
  }
  buildHelpers() {
    this.clear(this.helpers);
    if (this.debug.grid) {
      const grid = new THREE.GridHelper(60, 60, 0x8fa6b2, 0x40525c);
      grid.rotation.x = Math.PI / 2;
      this.helpers.add(grid);
    }
    if (this.debug.axes) this.helpers.add(new THREE.AxesHelper(1.5));
  }
  // The MuJoCo contact geometry is the extruded prisms, not the rendered road.
  buildCollisionView() {
    this.clear(this.collisionView);
    const material = new THREE.MeshBasicMaterial({
      color: 0x6ef2c0,
      wireframe: true,
      transparent: true,
      opacity: 0.55,
    });
    for (const tile of collisionTiles(this.map)) {
      const g = new THREE.BufferGeometry();
      g.setAttribute(
        'position',
        new THREE.Float32BufferAttribute(tile.vertices, 3),
      );
      g.setIndex(tile.faces);
      this.collisionView.add(new THREE.Mesh(g, material));
    }
  }
  forEachRobotMaterial(
    fn: (m: THREE.MeshStandardMaterial, geom: RobotGeom, skin: string) => void,
  ) {
    for (const g of this.model) {
      const mesh = this.robotMeshes.get(g.id);
      if (mesh) fn(mesh.material as THREE.MeshStandardMaterial, g, this.skin);
    }
    for (const remote of this.remotes.values())
      for (const child of remote.children)
        if (child instanceof THREE.Mesh && child.userData.geom)
          fn(
            child.material as THREE.MeshStandardMaterial,
            child.userData.geom,
            remote.userData.skin,
          );
  }
  repaintRobots() {
    this.forEachRobotMaterial((m, g, skin) => {
      if (g.textured) this.paintRobot(m as THREE.MeshStandardMaterial, g, skin);
    });
  }
  applyRobotTuning() {
    this.repaintRobots();
  }
  setChannel(channel: TextureChannel, on: boolean) {
    this.channels[channel] = on;
    this.repaintRobots();
  }
  // Swaps one robot texture for a locally chosen image without re-exporting assets.
  async setRobotTexture(channel: TextureChannel, file: File | null) {
    const srgb = channel === 'map' || channel === 'emissive';
    // A replacement is authored against the same atlas, so it uploads the same way.
    const custom = file
      ? this.robotAtlas(await this.loadCustomTexture(file, srgb), srgb)
      : null;
    if (channel === 'map') {
      // Overrides all eight skins at once; clearing it restores each atlas.
      const previous = this.customMap;
      this.customMap = custom;
      if (previous && this.customTextures.delete(previous)) previous.dispose();
    } else {
      // Dropping the entry is the restore: the shipped map is fetched again the
      // next time the channel is switched on.
      const previous = this.extraTextures.get(channel);
      if (custom) this.extraTextures.set(channel, custom);
      else this.extraTextures.delete(channel);
      if (previous) {
        this.customTextures.delete(previous);
        previous.dispose();
      }
    }
    this.setChannel(channel, this.channels[channel]);
  }
  async loadCustomTexture(file: File, srgb: boolean) {
    const url = URL.createObjectURL(file);
    try {
      const texture = await this.loadTexture(url, srgb);
      this.customTextures.add(texture);
      return texture;
    } finally {
      URL.revokeObjectURL(url);
    }
  }
  loadTexture(url: string, srgb: boolean) {
    return new Promise<THREE.Texture<HTMLImageElement>>((resolve, reject) => {
      new THREE.TextureLoader().load(
        url,
        (t) => {
          if (srgb) t.colorSpace = THREE.SRGBColorSpace;
          resolve(t);
        },
        undefined,
        () => reject(Error(`Texture failed to load: ${url}`)),
      );
    });
  }
  // Road UVs are in metres, so repeat is "tiles per metre" on each axis.
  setRoadTexture(
    texture: THREE.Texture | null,
    repeatU = this.roadTexture?.repeat.x ?? 1,
    repeatV = this.roadTexture?.repeat.y ?? 1,
  ) {
    if (this.roadTexture && this.roadTexture !== texture)
      this.roadTexture.dispose();
    this.roadTexture = texture;
    if (texture) {
      texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
      texture.repeat.set(repeatU, repeatV);
      texture.colorSpace = THREE.SRGBColorSpace;
    }
    const materials = this.roadMesh?.material;
    if (!Array.isArray(materials)) return;
    const base = materials[0] as THREE.MeshStandardMaterial;
    base.map = texture;
    // clear() leaves flagged materials' maps alone, so the next setMap keeps it.
    base.userData.robotTexture = !!texture;
    base.needsUpdate = true;
  }
  async loadRoadTexture(file: File, repeatU: number, repeatV: number) {
    const url = URL.createObjectURL(file);
    try {
      this.setRoadTexture(await this.loadTexture(url, true), repeatU, repeatV);
    } finally {
      URL.revokeObjectURL(url);
    }
  }
  stats() {
    const info = this.renderer.info;
    return {
      fps: this.fps,
      triangles: info.render.triangles,
      calls: info.render.calls,
      geometries: info.memory.geometries,
      textures: info.memory.textures,
      effects: this.effects.stats(),
    };
  }
  setEffectsQuality(value: EffectsQuality) {
    const level = effectsQuality(value),
      budget = EFFECT_LEVELS[level];
    this.effects.setQuality(level);
    this.renderer.setPixelRatio(
      Math.min(window.devicePixelRatio || 1, budget.dpr),
    );
    this.effects.resize(
      this.element.clientHeight * this.renderer.getPixelRatio(),
    );
    this.renderer.shadowMap.enabled = this.debug.shadows && budget.shadow > 0;
    if (this.sun.shadow.mapSize.x !== budget.shadow) {
      this.sun.shadow.map?.dispose();
      this.sun.shadow.map = null;
      this.sun.shadow.mapSize.setScalar(Math.max(1, budget.shadow));
      this.sun.shadow.needsUpdate = true;
    }
    this.renderer.domElement.dataset.effectsQuality = level;
    if (this.room) this.effects.syncRoom(this.room);
  }
  setModel(geoms: RobotGeometry[]) {
    this.effects.clear();
    this.frameBuffer.clear();
    this.frame = null;
    this.clear(this.robot);
    this.robotMeshes.clear();
    this.model = geoms;
    for (const g of geoms) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        'position',
        new THREE.BufferAttribute(g.vertices, 3),
      );
      geometry.setIndex(new THREE.BufferAttribute(g.faces, 1));
      if (g.normals?.length)
        geometry.setAttribute(
          'normal',
          new THREE.BufferAttribute(g.normals, 3),
        );
      else geometry.computeVertexNormals();
      if (g.uv?.length) {
        geometry.setAttribute('uv', new THREE.BufferAttribute(g.uv, 2));
        geometry.setAttribute('uv1', new THREE.BufferAttribute(g.uv, 2));
      }
      const mesh = new THREE.Mesh(geometry, this.robotMaterial(g, this.skin));
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.robot.add(mesh);
      this.robotMeshes.set(g.id, mesh);
    }
    this.ready = true;
    for (const r of this.remotes.values()) {
      this.clear(r);
      this.scene.remove(r);
    }
    this.remotes.clear();
    if (this.room) this.updateRoom(this.room, this.selfId);
    this.applyDebugMaterials();
  }
  setSkin(id: string) {
    this.skin = getSkin(id).id;
    this.repaintRobots();
  }
  setFrame(frame: PhysicsFrame) {
    const poses = new Map<number, { p: THREE.Vector3; q: THREE.Quaternion }>();
    const mat = new THREE.Matrix4();
    for (const id of this.robotMeshes.keys()) {
      const index = frame.geomIds ? frame.geomIds.indexOf(id) : id;
      if (index < 0) continue;
      const p = new THREE.Vector3().fromArray(frame.geomPos, index * 3);
      const a = frame.geomMat,
        k = index * 9;
      mat.set(
        a[k],
        a[k + 1],
        a[k + 2],
        0,
        a[k + 3],
        a[k + 4],
        a[k + 5],
        0,
        a[k + 6],
        a[k + 7],
        a[k + 8],
        0,
        0,
        0,
        0,
        1,
      );
      poses.set(id, {
        p,
        q: new THREE.Quaternion().setFromRotationMatrix(mat),
      });
    }
    this.frameBuffer.push({
      time: frame.wallTime - performance.timeOrigin,
      epoch: frame.epoch,
      poses,
      p: new THREE.Vector3(...(frame.p as [number, number, number])),
      q: new THREE.Quaternion(frame.q[1], frame.q[2], frame.q[3], frame.q[0]),
    });
  }
  setPlayerNames(visible: boolean) {
    this.showPlayerNames = visible;
    for (const remote of this.remotes.values())
      remote.userData.label.visible = visible;
  }
  makeRemote(id: string, name: string, skinId: string) {
    const g = new THREE.Group();
    this.scene.add(g);
    const skin = getSkin(skinId);
    for (const geom of this.model) {
      const source = this.robotMeshes.get(geom.id)!;
      const mesh = new THREE.Mesh(
        source.geometry.clone(),
        this.robotMaterial(geom, skinId),
      );
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.userData.geom = geom;
      g.add(mesh);
    }
    const label = this.label(name, skin.primary);
    label.position.set(0, 0, 0.34);
    label.scale.set(0.3, 0.094, 1);
    label.visible = this.showPlayerNames;
    g.add(label);
    g.userData = { name, label, skin: skinId };
    this.remotes.set(id, g);
    this.applyDebugMaterials();
    return g;
  }
  updateRoom(room: Room | null, id: string) {
    this.room = room ? { ...room, clockOffset: room.now - Date.now() } : null;
    this.selfId = id;
    this.effects.syncRoom(room);
    const me = room?.players.find((p) => p.id === id);
    if (me?.skin && me.skin !== this.skin) this.setSkin(me.skin);
    const live = new Set<string>();
    for (const p of room?.players ?? []) {
      if (p.id === id) continue;
      live.add(p.id);
      let r = this.remotes.get(p.id);
      if (r && (r.userData.name !== p.name || r.userData.skin !== p.skin)) {
        this.clear(r);
        this.scene.remove(r);
        this.remotes.delete(p.id);
        r = undefined;
      }
      r ??= this.makeRemote(p.id, p.name, p.skin);
      r.userData.target = p;
      r.visible = p.connected;
    }
    for (const [key, r] of this.remotes)
      if (!live.has(key)) {
        this.clear(r);
        this.scene.remove(r);
        this.remotes.delete(key);
      }
  }
  effectPosition = (id: string, target: THREE.Vector3) => {
    if (id === this.selfId && this.frame) {
      target.fromArray(this.frame.p);
      return true;
    }
    const remote = this.remotes.get(id);
    if (remote) {
      target.copy(remote.position);
      return true;
    }
    return false;
  };
  animate = () => {
    this.animation = requestAnimationFrame(this.animate);
    this.renderAt(performance.now());
  };
  renderAt(renderNow: number, wallNow = Date.now(), beforeDraw?: () => void) {
    const dt = Math.min(0.1, (renderNow - this.lastRender) / 1000);
    this.lastRender = renderNow;
    if (dt > 0) this.fps += ((1 / dt - this.fps) * dt) / 0.25;
    const blend = (at60: number) => 1 - Math.pow(1 - at60, dt * 60);
    const sample = this.frameBuffer.sample(renderNow);
    if (sample) {
      const { a, b, alpha } = sample;
      this.renderPosition.lerpVectors(a.p, b.p, alpha);
      this.renderQuaternion.slerpQuaternions(a.q, b.q, alpha);
      this.frame = {
        p: this.renderPosition.toArray(),
        q: [
          this.renderQuaternion.w,
          this.renderQuaternion.x,
          this.renderQuaternion.y,
          this.renderQuaternion.z,
        ],
      };
      for (const [id, mesh] of this.robotMeshes) {
        const pa = a.poses.get(id),
          pb = b.poses.get(id);
        if (!pa || !pb) continue;
        mesh.position.lerpVectors(pa.p, pb.p, alpha);
        mesh.quaternion.slerpQuaternions(pa.q, pb.q, alpha);
      }
    }
    const t = (renderNow - this.started) / 1000;
    const now = wallNow + (this.room?.clockOffset ?? 0);
    for (let i = 0; i < this.boxes.length; i++) {
      const b = this.boxes[i];
      const box = (this.room?.pickups ?? this.previewPickups)[i];
      b.visible =
        !!box?.p &&
        box.active !== false &&
        (box.availableAt ?? 0) <= now &&
        (box.expiresAt ?? Infinity) > now;
      if (!b.visible) continue;
      b.position.fromArray(box.p);
      const enabled = this.effects.quality !== 'off';
      const [gem, halo] = b.children as THREE.Mesh[];
      const color = box.color ?? '#ffe66d';
      (gem.material as THREE.MeshStandardMaterial).emissive.set(color);
      (halo.material as THREE.MeshBasicMaterial).color.set(color);
      halo.visible = enabled;
      const age = Math.min(
        1,
        Math.max(0, (now - (box.availableAt ?? 0)) / 350),
      );
      b.scale.setScalar(
        (enabled ? 1 - (1 - age) ** 3 : 1) * (box.value === 3 ? 1.2 : 1),
      );
      if (enabled) {
        gem.rotation.set(0.15 * Math.sin(t * 2 + i), 0, t * 1.8 + i);
        gem.position.z = Math.sin(t * 3 + i) * 0.045;
        halo.scale.setScalar(1 + 0.15 * Math.sin(t * 4 + i));
      } else {
        gem.rotation.set(0, 0, 0);
        gem.position.z = 0;
      }
    }
    for (const r of this.remotes.values()) {
      const p = r.userData.target;
      if (p?.p) {
        r.position.lerp(
          new THREE.Vector3(...(p.p as [number, number, number])),
          0.3,
        );
        r.quaternion.slerp(
          new THREE.Quaternion(p.q[1], p.q[2], p.q[3], p.q[0]),
          0.3,
        );
        for (const child of r.children) {
          if (!child.userData.geom) continue;
          const pose = avatarMatrix(child.userData.geom, p.j ?? []);
          pose.decompose(child.position, child.quaternion, child.scale);
        }
      }
    }
    if (this.debug.freeCamera) {
      this.debugCamera?.update(dt);
    } else if (this.preview) {
      const orbit = t * 0.06;
      const bounds = getTrack(this.map).bounds;
      const center = new THREE.Vector3(
        (bounds.minX + bounds.maxX) / 2,
        (bounds.minY + bounds.maxY) / 2,
        0.5,
      );
      const radius =
        Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY) * 0.9;
      const desired = center
        .clone()
        .add(
          new THREE.Vector3(
            -radius * Math.cos(orbit),
            -radius * 0.85 * Math.cos(orbit * 0.7),
            radius * 0.95,
          ),
        );
      this.camera.position.lerp(desired, 0.025);
      this.target.lerp(center, 0.04);
    } else if (this.frame) {
      const p = this.frame.p,
        q = this.frame.q,
        yaw = Math.atan2(
          2 * (q[0] * q[3] + q[1] * q[2]),
          1 - 2 * (q[2] * q[2] + q[3] * q[3]),
        );
      // A close, centered chase view fits beneath the 0.8 m tunnel clearance.
      const target = new THREE.Vector3(p[0], p[1], Math.max(p[2], -0.1) + 0.12);
      const desired = target
        .clone()
        .add(
          new THREE.Vector3(-1.1 * Math.cos(yaw), -1.1 * Math.sin(yaw), 0.38),
        );
      this.camera.position.lerp(desired, blend(0.09));
      this.target.lerp(target, blend(0.16));
    }
    this.camera.lookAt(this.target);
    beforeDraw?.();
    this.effects.update(dt, now, this.effectPosition);
    this.renderer.render(this.scene, this.camera);
  }
  dispose() {
    cancelAnimationFrame(this.animation);
    this.resize.disconnect();
    this.clear(this.world);
    this.clear(this.robot);
    this.effects.dispose();
    this.clear(this.helpers);
    this.clear(this.collisionView);
    for (const r of this.remotes.values()) this.clear(r);
    for (const t of this.extraTextures.values()) t.dispose();
    this.extraTextures.clear();
    for (const t of this.skinMaps.values()) t.dispose();
    this.skinMaps.clear();
    for (const t of this.customTextures) t.dispose();
    this.customTextures.clear();
    this.roadTexture?.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
