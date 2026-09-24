import type { ReactNode } from 'react';
import './globals.css';
export const metadata = {
  title: 'OPEN BENI · BENI GAME',
  description: 'Multiplayer BENI racing powered by MuJoCo and ONNX',
};
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
