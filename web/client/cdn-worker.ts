import loaders from '../.generated/loaders.json';
/** HF serves JS as plain text; activate its cached source as a typed Blob. */
export function createSimulationWorker(path: string, name: string) {
  const base = (window as Window & { __FRONTEND_ASSET_BASE__?: string })
    .__FRONTEND_ASSET_BASE__;
  if (new URL(location.href).searchParams.get('assetSource') === 'local')
    name += ':local';
  const local = new URL(path, location.origin).href;
  if (!base || name.endsWith(':local'))
    return new Worker(local, { type: 'module', name });
  const config = JSON.stringify({ url: base + path });
  const url = URL.createObjectURL(
    new Blob([`globalThis.__OPENBENI_BOOT__=${config};\n`, loaders.worker], {
      type: 'text/javascript',
    }),
  );
  try {
    return new Worker(url, { type: 'module', name });
  } finally {
    URL.revokeObjectURL(url);
  }
}
