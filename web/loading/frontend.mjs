import { download } from './download.mjs';
import { cached, remember } from './cache.mjs';
const config = globalThis.__OPENBENI_BOOT__;
window.__FRONTEND_ASSET_BASE__ = config.base;
async function source(path) {
  const url = config.base + path;
  const hit = await cached('openbeni-code-v1', url);
  if (hit) return hit.text();
  const response = await download(url);
  await remember('openbeni-code-v1', url, response);
  return response.text();
}
try {
  const [js, ...styles] = await Promise.all(
    [...config.scripts, ...config.styles].map(source),
  );
  for (const css of styles) {
    const style = document.createElement('style');
    style.textContent = css;
    document.head.append(style);
  }
  const url = URL.createObjectURL(new Blob([js], { type: 'text/javascript' }));
  try {
    await import(/* @vite-ignore */ url);
  } finally {
    URL.revokeObjectURL(url);
  }
  document.getElementById('boot')?.remove();
} catch (error) {
  console.error('Game startup failed', error);
  const boot = document.getElementById('boot');
  boot.textContent = 'Unable to download game. ';
  for (const [label, backup] of [
    ['Retry download', false],
    ['Try backup source (may be slower)', true],
  ]) {
    const button = document.createElement('button');
    button.textContent = label;
    button.onclick = () => {
      const url = new URL(location.href);
      if (backup) url.searchParams.set('assetSource', 'local');
      location.assign(url.href);
    };
    boot.append(button);
  }
}
