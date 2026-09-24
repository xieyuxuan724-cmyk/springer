import loaders from '../.generated/loaders.json' with { type: 'json' };
/** HF resolve URLs serve scripts as plain text: load them as typed Blob modules. */
export function frontendPage(html, base) {
  if (!base) return html;
  const scripts = [
    ...html.matchAll(/<script\b[^>]*src="([^"]+)"[^>]*><\/script>/g),
  ].map((m) => m[1]);
  const styles = [
    ...html.matchAll(/<link\b[^>]*rel="stylesheet"[^>]*href="([^"]+)"[^>]*>/g),
  ].map((m) => m[1]);
  if (scripts.length !== 1 || !scripts[0].startsWith('/assets/'))
    throw Error('Unsupported frontend entry');
  const config = JSON.stringify({
    base: base.replace(/\/$/, ''),
    scripts,
    styles,
  }).replace(/</g, '\\u003c');
  const loader = `<div id="boot" role="status" style="position:fixed;inset:0;display:grid;place-content:center;background:#11181a;color:white;font:16px system-ui">Loading game…</div><script type="module">
globalThis.__OPENBENI_BOOT__=${config};
${loaders.frontend.replace(/<\/script/gi, '<\\/script')}
</script>`;
  return html
    .replace(/<script\b[^>]*src="[^"]+"[^>]*><\/script>/g, '')
    .replace(/<link\b[^>]*rel="stylesheet"[^>]*>/g, '')
    .replace('</body>', loader + '</body>');
}
