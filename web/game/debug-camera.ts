import * as THREE from 'three';
// Z-up orbit / pan / dolly controller for the debug view. Pointer gestures are
// always live; keyboard flight is opt-in because the game also drives on WASD.
const Z = new THREE.Vector3(0, 0, 1);
const MIN_RADIUS = 0.25,
  MAX_RADIUS = 220,
  MIN_POLAR = 0.02,
  MAX_POLAR = Math.PI - 0.02;
type Pointer = { id: number; x: number; y: number; button: number };
export class DebugCamera {
  radius = 8;
  azimuth = -Math.PI / 2;
  polar = 1.1;
  fly = false;
  speed = 6;
  private keys = new Set<string>();
  private pointers = new Map<number, Pointer>();
  private detach: () => void;
  constructor(
    private camera: THREE.PerspectiveCamera,
    private dom: HTMLElement,
    private target: THREE.Vector3,
  ) {
    this.syncFromCamera();
    const down = (e: PointerEvent) => {
      if (e.button > 2) return;
      dom.setPointerCapture(e.pointerId);
      this.pointers.set(e.pointerId, {
        id: e.pointerId,
        x: e.clientX,
        y: e.clientY,
        button: e.button,
      });
    };
    const move = (e: PointerEvent) => {
      const p = this.pointers.get(e.pointerId);
      if (!p) return;
      const dx = e.clientX - p.x,
        dy = e.clientY - p.y;
      p.x = e.clientX;
      p.y = e.clientY;
      // Two fingers or any non-primary button pans; a single primary drag orbits.
      if (p.button === 0 && this.pointers.size === 1) {
        this.azimuth -= dx * 0.005;
        this.polar = THREE.MathUtils.clamp(
          this.polar - dy * 0.005,
          MIN_POLAR,
          MAX_POLAR,
        );
      } else {
        const scale = this.radius * 0.0018;
        this.pan(-dx * scale, dy * scale);
      }
      this.place();
    };
    const up = (e: PointerEvent) => {
      this.pointers.delete(e.pointerId);
      if (dom.hasPointerCapture(e.pointerId))
        dom.releasePointerCapture(e.pointerId);
    };
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      this.radius = THREE.MathUtils.clamp(
        this.radius * Math.exp(e.deltaY * 0.0012),
        MIN_RADIUS,
        MAX_RADIUS,
      );
      this.place();
    };
    const menu = (e: Event) => e.preventDefault();
    const keyDown = (e: KeyboardEvent) => {
      if (
        (e.target as HTMLElement)?.closest(
          'input,textarea,select,[contenteditable=true]',
        )
      )
        return;
      this.keys.add(e.code);
    };
    const keyUp = (e: KeyboardEvent) => this.keys.delete(e.code);
    const blur = () => this.keys.clear();
    dom.addEventListener('pointerdown', down);
    dom.addEventListener('pointermove', move);
    dom.addEventListener('pointerup', up);
    dom.addEventListener('pointercancel', up);
    dom.addEventListener('wheel', wheel, { passive: false });
    dom.addEventListener('contextmenu', menu);
    window.addEventListener('keydown', keyDown);
    window.addEventListener('keyup', keyUp);
    window.addEventListener('blur', blur);
    this.detach = () => {
      dom.removeEventListener('pointerdown', down);
      dom.removeEventListener('pointermove', move);
      dom.removeEventListener('pointerup', up);
      dom.removeEventListener('pointercancel', up);
      dom.removeEventListener('wheel', wheel);
      dom.removeEventListener('contextmenu', menu);
      window.removeEventListener('keydown', keyDown);
      window.removeEventListener('keyup', keyUp);
      window.removeEventListener('blur', blur);
    };
  }
  // Start from wherever the game camera left off so enabling is not a jump cut.
  syncFromCamera() {
    const offset = this.camera.position.clone().sub(this.target);
    this.radius = THREE.MathUtils.clamp(
      offset.length() || 8,
      MIN_RADIUS,
      MAX_RADIUS,
    );
    this.azimuth = Math.atan2(offset.y, offset.x);
    this.polar = THREE.MathUtils.clamp(
      Math.acos(THREE.MathUtils.clamp(offset.z / this.radius, -1, 1)),
      MIN_POLAR,
      MAX_POLAR,
    );
  }
  lookAt(point: THREE.Vector3, radius = this.radius) {
    this.target.copy(point);
    this.radius = THREE.MathUtils.clamp(radius, MIN_RADIUS, MAX_RADIUS);
    this.place();
  }
  // Ground-plane heading, so keyboard flight never sinks while looking down.
  private forward() {
    return this.target.clone().sub(this.camera.position).setZ(0).normalize();
  }
  // Panning follows the screen, so it uses the true view basis instead.
  private pan(right: number, up: number) {
    const view = this.target.clone().sub(this.camera.position).normalize(),
      r = view.clone().cross(Z).normalize(),
      u = r.clone().cross(view).normalize();
    this.target.addScaledVector(r, right).addScaledVector(u, up);
  }
  private place() {
    const s = Math.sin(this.polar);
    this.camera.position.set(
      this.target.x + this.radius * s * Math.cos(this.azimuth),
      this.target.y + this.radius * s * Math.sin(this.azimuth),
      this.target.z + this.radius * Math.cos(this.polar),
    );
  }
  update(dt: number) {
    if (this.fly && this.keys.size) {
      const k = this.keys,
        f = this.forward(),
        r = f.clone().cross(Z).normalize();
      const step =
        this.speed * dt * (k.has('ShiftLeft') || k.has('ShiftRight') ? 3 : 1);
      const ahead = (k.has('KeyW') ? 1 : 0) - (k.has('KeyS') ? 1 : 0),
        side = (k.has('KeyD') ? 1 : 0) - (k.has('KeyA') ? 1 : 0),
        lift = (k.has('KeyE') ? 1 : 0) - (k.has('KeyQ') ? 1 : 0);
      if (ahead || side || lift) {
        this.target
          .addScaledVector(f, ahead * step)
          .addScaledVector(r, side * step);
        this.target.z += lift * step;
      }
    }
    this.place();
  }
  dispose() {
    this.detach();
    this.keys.clear();
    this.pointers.clear();
  }
}
