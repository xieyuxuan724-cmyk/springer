import { Button } from '@/components/ui/button';
import { useGame } from './context';
import { Input } from '@/components/ui/input';
import { Wifi, WifiOff, LogOut } from 'lucide-react';
import { useNameForm } from './context';
import { SettingsMenu } from './SettingsMenu';
export function GameHeader() {
  const {
    room,
    selected,
    name,
    connected,
    ping,
    racing,
    session,
    mode,
    connectionSleeping,
  } = useGame();
  const { draftName, setDraftName, editing, setEditing, submitName } =
    useNameForm();
  return (
    <header className="topbar">
      <a className="brand" href="/">
        OPEN<span>BENI</span>
      </a>
      {room && <span className="current-course">{selected.name}</span>}
      <div className="header-right">
        {name &&
          (editing ? (
            <form onSubmit={submitName} className="rename">
              <Input
                aria-label="Rename driver"
                value={draftName}
                onChange={(e) => setDraftName(e.target.value)}
                maxLength={40}
              />
              <Button type="submit" disabled={!connected}>
                Save
              </Button>
              <Button
                variant="ghost"
                onClick={() => {
                  setDraftName(name);
                  setEditing(false);
                }}
              >
                Cancel
              </Button>
            </form>
          ) : (
            <Button
              variant="ghost"
              className="header-user"
              title={`Rename driver: ${draftName || name}`}
              onClick={() => setEditing(true)}
            >
              <span className="avatar">{(draftName || name).slice(0, 1)}</span>
              <span className="header-name">{draftName || name}</span>
            </Button>
          ))}
        {name && mode === 'multiplayer' && (
          <span
            className={`status ${connected ? '' : 'offline'}`}
            title={
              connected
                ? `Connected · ${ping} ms`
                : connectionSleeping
                  ? 'Disconnected'
                  : 'Connecting…'
            }
            role="status"
          >
            {connected ? <Wifi size={15} /> : <WifiOff size={15} />}
            <span className={connected ? 'sr-only' : ''}>
              {connected
                ? 'Connected'
                : connectionSleeping
                  ? 'Disconnected'
                  : 'Connecting…'}
            </span>
          </span>
        )}
        <SettingsMenu />
        {room && (
          <Button
            variant="ghost"
            className="header-exit"
            aria-label={racing ? 'Leave race' : 'Leave room'}
            title="Leave the room and return to the lobby"
            onClick={session.leaveRoom}
            onPointerDown={(event) => {
              // A second finger must be able to exit while the first holds the stick.
              if (event.pointerType === 'touch' && event.button === 0) {
                event.preventDefault();
                session.leaveRoom();
              }
            }}
          >
            <LogOut size={15} />
            <span>Leave</span>
          </Button>
        )}
      </div>
    </header>
  );
}
