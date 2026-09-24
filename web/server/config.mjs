/** Resource limits are server-owned and validated before opening a port. */
export function serverLimits(env = process.env) {
  const integer = (key, fallback, minimum = 1) => {
    const value = env[key] === undefined ? fallback : Number(env[key]);
    if (
      !Number.isSafeInteger(value) ||
      value < minimum ||
      env[key]?.trim() === ''
    )
      throw Error(`${key} must be an integer >= ${minimum}`);
    return value;
  };
  return {
    maxRooms: integer('MAX_ROOMS', 10, 0),
    maxConnections: integer('MAX_CONNECTIONS', 256),
  };
}
