import type { WorkerCommand, WorkerEvent } from '../shared/protocol.ts';
export type WorkerPort = Pick<
  Worker,
  'postMessage' | 'terminate' | 'onmessage' | 'onerror'
>;
/** Owns exactly one worker generation. Retired workers and file reads cannot publish. */
export class SimulationClient {
  private worker: WorkerPort | null = null;
  private generation = 0;
  private timer: ReturnType<typeof setTimeout> | undefined;
  private createWorker: () => WorkerPort;
  private receive: (event: WorkerEvent) => void;
  constructor(
    createWorker: () => WorkerPort,
    receive: (event: WorkerEvent) => void,
  ) {
    this.createWorker = createWorker;
    this.receive = receive;
  }
  start() {
    this.dispose();
    const generation = this.generation;
    const worker = this.createWorker();
    this.worker = worker;
    worker.onmessage = ({ data }: MessageEvent<WorkerEvent>) => {
      if (generation !== this.generation) return;
      if (data.type === 'ready') this.clearTimer();
      if (data.type === 'fatal') this.dispose();
      this.receive(data);
    };
    worker.onerror = (event) => {
      if (generation === this.generation)
        this.fail(`Physics engine failed to start: ${event.message}`);
    };
  }
  post = (message: WorkerCommand, transfer: Transferable[] = []) => {
    this.worker?.postMessage(message, transfer);
  };
  async upload(file: Pick<File, 'name' | 'arrayBuffer'>) {
    const generation = this.generation;
    this.clearTimer();
    this.timer = setTimeout(() => {
      if (generation === this.generation)
        this.fail(
          'Policy loading took more than 20 seconds. Restore the default policy and try again.',
        );
    }, 20000);
    try {
      const bytes = await file.arrayBuffer();
      if (generation !== this.generation || !this.worker) return;
      this.post({ type: 'policy', bytes, label: file.name }, [bytes]);
    } catch (error) {
      if (generation === this.generation)
        this.fail(`Unable to read policy file: ${String(error)}`);
    }
  }
  private clearTimer() {
    clearTimeout(this.timer);
    this.timer = undefined;
  }
  private fail(text: string) {
    this.dispose();
    this.receive({ type: 'fatal', text });
  }
  dispose() {
    ++this.generation;
    this.clearTimer();
    if (this.worker) {
      this.worker.onmessage = null;
      this.worker.onerror = null;
      this.worker.terminate();
      this.worker = null;
    }
  }
}
