// Command contract from python/beni_rl/jump.py, with player-controlled timing.
// 40-input policies keep manual timing and drive holds; 42-input policies brake yaw.
export const BENI_JUMP_REFERENCE = Object.freeze({ frames: 41, height: 0.18 });
export class JumpController {
  constructor(reference = BENI_JUMP_REFERENCE) {
    this.reference = reference;
    this.reset();
  }
  reset() {
    this.frame = -1;
    this.readySteps = 0;
    this.tracking = false;
    this.age = 0;
    this.airSteps = 0;
    this.lifted = false;
    this.landed = false;
    this.landAge = -1;
    this.readyHold = 0;
    this.heldDrive = null;
    this.preparing = false;
    this.prepareSteps = this.yawReadySteps = 0;
    this.headingActive = false;
    this.headingReference = 0;
  }
  request(input = { vx: 0, wz: 0 }) {
    if (this.headingEnabled) {
      if (!input.ready || this.tracking) return false;
      this.headingReference = input.heading;
      this.headingActive = this.preparing = true;
      this.prepareSteps = this.yawReadySteps = 0;
    }
    // A 40-input policy may restart an active jump.
    this.frame = this.preparing ? -1 : 0;
    this.tracking = true;
    this.age = this.airSteps = this.readyHold = 0;
    this.lifted = this.landed = false;
    this.landAge = -1;
    // 42-input policies use commandHold; keep the 40-input landing drive hold.
    this.heldDrive = this.headingEnabled ? null : { vx: input.vx ?? 0, wz: 0 };
    return true;
  }
  get commandHold() {
    return this.tracking && (this.preparing ||
      (!this.landed && this.age < 200) ||
      (this.landed && this.readyHold < 100 && this.age < this.landAge+150));
  }
  headingObservation(heading) {
    const delta = heading - this.headingReference;
    return this.headingActive ? [Math.atan2(Math.sin(delta), Math.cos(delta))/Math.PI, 1] : [0, 0];
  }
  get command() {
    return this.frame < 0
      ? [0, 0, 0, 0]
      : [1, this.frame / 40, this.reference.height, 0];
  }
  update({ ready, air, contact, body, height, tilt, wz }) {
    this.readySteps = ready ? this.readySteps + 1 : 0;
    if (this.headingEnabled) {
      if (body && (tilt >= Math.PI/3 || height < .065)) this.headingActive = false;
      if (this.preparing) {
        this.yawReadySteps = ready && Math.abs(wz) <= .3 ? this.yawReadySteps + 1 : 0;
        if (++this.prepareSteps >= 200 || body) {
          this.preparing = this.tracking = this.headingActive = false;
        } else if (this.yawReadySteps >= 20) {
          this.preparing = false;
          this.frame = 0;
        }
        return;
      }
    }
    if (this.frame >= 0 && ++this.frame >= this.reference.frames)
      this.frame = -1;
    if (!this.tracking) return;
    this.age++;
    this.airSteps = air ? this.airSteps + 1 : 0;
    this.lifted ||= this.airSteps >= 5;
    if (!this.landed && this.lifted && contact) {
      this.landed = true;
      this.landAge = this.age;
    }
    // Stop observing after a stable recovery or timeout (100 Hz policy steps).
    // This observation state does not block a 40-input policy's manual request.
    this.readyHold =
      ready && contact && !air && (this.landed || this.frame < 0)
        ? this.readyHold + 1
        : 0;
    // Allow the latest player input back after 150 ms of balanced wheel support.
    // A missed landing cannot keep controls held indefinitely (1.2 s maximum).
    if ((this.frame < 0 && this.readyHold >= 15) || this.age >= 120)
      this.heldDrive = null;
    if ((this.frame < 0 && this.readyHold >= 100) || this.age >= 900)
      this.tracking = false;
  }
}
