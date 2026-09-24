'use client';
import { GameProvider, GameCanvas, useGame } from './game/context';
import { GameHeader } from './game/GameHeader';
import { LoginPanel } from './game/LoginPanel';
import { GaragePanel } from './game/GaragePanel';
import { RoomPanel } from './game/RoomPanel';
import { RaceOverlay } from './game/RaceOverlay';
import { DriveControls } from './game/DriveControls';
import { Minimap } from './game/Minimap';
import { GameFeedback } from './game/GameFeedback';
function GameScreen() {
  const { name, room, racing } = useGame();
  return (
    <main className={`game-shell ${racing ? 'is-racing' : ''}`}>
      <GameHeader />
      <section className="viewport">
        <GameCanvas />
        {!name ? (
          <LoginPanel />
        ) : !room ? (
          <GaragePanel />
        ) : (
          <>
            <RoomPanel />
            <RaceOverlay />
            <DriveControls />
            <Minimap />
          </>
        )}
        <GameFeedback />
      </section>
    </main>
  );
}
export default function Home() {
  return (
    <GameProvider>
      <GameScreen />
    </GameProvider>
  );
}
