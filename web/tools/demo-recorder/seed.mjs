// Repeatable pickup placement/bonus rolls for physical regression runs.
// Live rooms continue to use RaceService's cryptographic random source.
export function seededRandom(seed) {
  if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff)
    throw Error('Seed must be an unsigned 32-bit integer');
  let state = seed;
  return () => (state = (1664525 * state + 1013904223) >>> 0) / 2 ** 32;
}
