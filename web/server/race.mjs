import { randomBytes, randomUUID, randomInt } from 'node:crypto';
import { RaceRules } from '../shared/race-rules.mjs';
export { cleanName } from '../shared/validation.mjs';
export class RaceService extends RaceRules {
  constructor(options = {}) {
    super({
      id: randomUUID,
      token: () => randomBytes(24).toString('hex'),
      roomCode: () => randomBytes(3).toString('hex').slice(0, 5).toUpperCase(),
      random: () => randomInt(0x1000000) / 0x1000000,
      ...options,
    });
  }
}
