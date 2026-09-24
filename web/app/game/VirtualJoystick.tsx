import { useEffect, useRef, useSyncExternalStore } from 'react';
import { Move } from 'lucide-react';
import { useSession } from './context';

export function VirtualJoystick() {
  const session = useSession();
  const { input } = session;
  const stick = useSyncExternalStore(
    input.subscribeStick,
    input.getStick,
    input.getStick,
  );
  const pad = useRef<HTMLButtonElement>(null);
  const pointer = useRef<number | null>(null);

  useEffect(() => {
    const element = pad.current;
    const releaseCapture = () => {
      const id = pointer.current;
      pointer.current = null;
      if (id !== null && element?.hasPointerCapture(id))
        element.releasePointerCapture(id);
    };
    // Reset/disconnect/blur clear the input and release its captured finger too.
    const unsubscribe = input.subscribeStick(() => {
      if (!input.getStick()) releaseCapture();
    });
    const cancel = () => {
      releaseCapture();
      input.releaseStick();
    };
    window.addEventListener('resize', cancel);
    return () => {
      window.removeEventListener('resize', cancel);
      unsubscribe();
      cancel();
    };
  }, [input]);

  const move = (event: React.PointerEvent<HTMLButtonElement>) => {
    if (pointer.current !== event.pointerId) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const travel = bounds.width * 0.3;
    input.setStick(
      (event.clientX - bounds.x - bounds.width / 2) / travel,
      (event.clientY - bounds.y - bounds.height / 2) / travel,
    );
  };
  const release = (event: React.PointerEvent<HTMLButtonElement>) => {
    if (pointer.current !== event.pointerId) return;
    pointer.current = null;
    input.releaseStick();
    if (event.currentTarget.hasPointerCapture(event.pointerId))
      event.currentTarget.releasePointerCapture(event.pointerId);
  };
  return (
    <button
      type="button"
      ref={pad}
      className="virtual-joystick"
      aria-label="Steering joystick"
      aria-keyshortcuts="W A S D ArrowUp ArrowDown ArrowLeft ArrowRight Space"
      onKeyDown={(event) => {
        if (event.code === 'Space') {
          event.preventDefault();
          if (!event.repeat) session.post({ type: 'jump' });
        }
      }}
      title="Up/down to move forward/backward, left/right to steer"
      data-active={stick !== null}
      onPointerDown={(event) => {
        if (pointer.current !== null || event.button !== 0) return;
        event.preventDefault();
        pointer.current = event.pointerId;
        event.currentTarget.setPointerCapture(event.pointerId);
        move(event);
      }}
      onPointerMove={move}
      onPointerUp={release}
      onPointerCancel={release}
      onLostPointerCapture={release}
      onContextMenu={(event) => event.preventDefault()}
    >
      <span className="joystick-ring" />
      <span
        className="joystick-thumb"
        style={{
          left: `${50 + (stick?.x ?? 0) * 30}%`,
          top: `${50 + (stick?.y ?? 0) * 30}%`,
        }}
      >
        <Move size={19} aria-hidden="true" />
      </span>
    </button>
  );
}
