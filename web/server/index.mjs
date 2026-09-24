import { createGameServer } from './app.mjs';
const dev = process.argv.includes('--dev');
const host = process.env.HOST || '0.0.0.0',
  port = Number(process.env.PORT || 3000);
const app = await createGameServer({ dev });
try {
  await app.listen(port, host);
  console.log(
    `OPEN BENI listening on http://${host}:${port} (${dev ? 'development' : 'production'})`,
  );
} catch (error) {
  await app.close();
  throw error;
}
for (const signal of ['SIGINT', 'SIGTERM'])
  process.once(signal, () => {
    void app.close();
  });
