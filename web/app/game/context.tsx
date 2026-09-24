// Vite emits the worker URL for this resource query.
// oxlint-disable-next-line import/default
import workerUrl from '../../game/sim.worker?worker&url';
import { createSimulationWorker } from '../../client/cdn-worker';
import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from 'react';
import { Viewer } from '../../game/viewer';
import { readEffectsQuality } from '../../game/race-effects';
import { GameSession } from '../../client/game-session';
import { selectGame } from '../../client/selectors';
const Context = createContext<GameSession | null>(null);
export function GameProvider({ children }: { children: ReactNode }) {
  const [session] = useState(
    () =>
      new GameSession(
        {
          createWorker: (mode) =>
            createSimulationWorker(workerUrl, mode),
          createSocket: () =>
            new WebSocket(
              `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`,
            ),
          createViewer: (element, quality) => new Viewer(element, quality),
        },
        readEffectsQuality(),
      ),
  );
  return <Context.Provider value={session}>{children}</Context.Provider>;
}
export function useSession() {
  const session = useContext(Context);
  if (!session) throw Error('GameProvider is missing');
  return session;
}
export function useGame() {
  const session = useSession();
  const state = useSyncExternalStore(
    session.subscribe,
    session.getSnapshot,
    session.getSnapshot,
  );
  return useMemo(
    () => ({
      ...selectGame(state),
      session,
      send: session.send,
      post: session.post,
      upload: session.upload,
      restore: session.restore,
      alert: session.notify,
      setNotice: session.notify,
      changeMap: session.changeMap,
      getViewer: session.getViewer,
      setDebugOpen: session.setDebugOpen,
      setShowMinimap: session.setShowMinimap,
      setShowNames: session.setShowNames,
      setEffectLevel: session.setEffectLevel,
    }),
    [session, state],
  );
}
export function GameCanvas() {
  const canvas = useRef<HTMLDivElement>(null),
    session = useSession();
  useEffect(() => {
    session.start(canvas.current!);
    return () => session.dispose();
  }, [session]);
  return <div ref={canvas} className="canvas-host" />;
}
export function useNameForm() {
  const { name, session, alert } = useGame();
  const [draft, setDraft] = useState<{ name: string; value: string } | null>(
    null,
  );
  const [editing, setEditing] = useState(false);
  const draftName = draft?.name === name ? draft.value : name;
  const setDraftName = (value: string) => setDraft({ name, value });
  const submitName = (event: React.SubmitEvent<HTMLFormElement>) => {
    event.preventDefault();
    try {
      session.submitName(draftName);
      setEditing(false);
    } catch (error) {
      alert((error as Error).message);
    }
  };
  return { draftName, setDraftName, editing, setEditing, submitName };
}
export function useDisclosure() {
  const ref = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    const dismiss = (event: PointerEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node))
        ref.current.open = false;
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && ref.current?.open) {
        ref.current.open = false;
        ref.current.querySelector('summary')?.focus();
      }
    };
    document.addEventListener('pointerdown', dismiss);
    document.addEventListener('keydown', escape);
    return () => {
      document.removeEventListener('pointerdown', dismiss);
      document.removeEventListener('keydown', escape);
    };
  }, []);
  return {
    ref,
    close: () => {
      if (ref.current) ref.current.open = false;
    },
  };
}
