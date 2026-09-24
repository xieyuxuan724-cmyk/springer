/**
 * Shared by the page bootstrap, worker bootstrap and simulation downloads.
 * @param {string} url
 * @param {{signal?: AbortSignal, cache?: RequestCache, onProgress?: (bytes: number) => void,
 * validate?: (blob: Blob) => Promise<void>, idleMs?: number, attempts?: number, retryDelayMs?: number}} options
 */
export async function download(
  url,
  {
    signal,
    cache,
    onProgress = () => {},
    validate = async () => {},
    idleMs = 60_000,
    attempts = 3,
    retryDelayMs = 1000,
  } = {},
) {
  for (let attempt = 1; attempt <= attempts; attempt++) {
    signal?.throwIfAborted();
    const controller = new AbortController();
    const abort = () => controller.abort(signal.reason);
    signal?.addEventListener('abort', abort, { once: true });
    let timer;
    let reader;
    let received = 0;
    let retryable = true;
    const arm = () => {
      clearTimeout(timer);
      timer = setTimeout(
        () => controller.abort(new Error('Download stalled')),
        idleMs,
      );
    };
    onProgress(0);
    try {
      arm();
      const response = await fetch(url, { cache, signal: controller.signal });
      if (!response.ok) {
        retryable =
          response.status === 408 ||
          response.status === 429 ||
          response.status >= 500;
        await response.body?.cancel();
        throw Error(`HTTP ${response.status}`);
      }
      reader = response.body?.getReader();
      if (!reader) throw Error('Response body unavailable');
      const chunks = [];
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        if (!value.byteLength) continue;
        chunks.push(value);
        received += value.byteLength;
        arm();
        onProgress(received);
      }
      clearTimeout(timer);
      const blob = new Blob(chunks, {
        type:
          response.headers.get('Content-Type') || 'application/octet-stream',
      });
      await validate(blob);
      signal?.throwIfAborted();
      return new Response(blob);
    } catch (error) {
      if (signal?.aborted) throw signal.reason;
      console.warn('Download attempt failed', {
        url,
        attempt,
        received,
        error: String(error),
      });
      if (!retryable || attempt === attempts)
        throw Error(
          'Download failed. Please retry or choose the backup source.',
          { cause: error },
        );
    } finally {
      clearTimeout(timer);
      controller.abort();
      await reader?.cancel().catch(() => {});
      reader?.releaseLock();
      signal?.removeEventListener('abort', abort);
    }
    await new Promise((resolve, reject) => {
      signal?.throwIfAborted();
      const cancel = () => {
        clearTimeout(delay);
        reject(signal.reason);
      };
      const delay = setTimeout(() => {
        signal?.removeEventListener('abort', cancel);
        resolve();
      }, attempt * retryDelayMs);
      signal?.addEventListener('abort', cancel, { once: true });
    });
  }
}
