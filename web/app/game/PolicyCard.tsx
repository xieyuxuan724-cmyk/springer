import { Button } from '@/components/ui/button';
import { useGame } from './context';
import { useRef } from 'react';
import { Upload, RotateCcw } from 'lucide-react';
export function PolicyCard() {
  const { loaded, fatal, room, policyLabel, engineStatus, upload, restore } =
    useGame();
  const loading = !loaded && !fatal;
  const canChangePolicy = loaded && (!room || room.phase === 'lobby');
  const policyFile = useRef<HTMLInputElement>(null);
  return (
    <section
      className="policy-card"
      aria-label="Driving policy"
      aria-busy={loading}
    >
      <span className="panel-subheading">Driving policy</span>
      <strong
        className="policy-name"
        title={fatal || (loaded ? policyLabel : engineStatus)}
        role="status"
      >
        {loading && <span className="spinner" aria-hidden="true" />}
        <span>
          {fatal ? 'Loading failed' : loaded ? policyLabel : engineStatus}
        </span>
      </strong>
      <div className="policy-actions">
        <Button
          className="upload-policy"
          disabled={!canChangePolicy}
          title="Choose an ONNX file, up to 64 MB; loaded locally"
          onClick={() => policyFile.current?.click()}
        >
          <Upload size={15} />
          Change policy
        </Button>
        <Button
          variant="ghost"
          className="restore"
          onClick={restore}
          disabled={!canChangePolicy}
        >
          <RotateCcw size={14} />
          Restore default
        </Button>
      </div>
      <input
        ref={policyFile}
        type="file"
        accept=".onnx"
        aria-label="Upload an ONNX driving policy"
        hidden
        disabled={!canChangePolicy}
        onChange={(e) => {
          void upload(e.target.files?.[0]);
          e.target.value = '';
        }}
      />
      <details className="policy-requirements">
        <summary>ONNX requirements</summary>
        <p>
          BENI compatible · 40 or 42 observations · 7 actions · Up to 64 MB · Local
          only
        </p>
      </details>
      {room && room.phase !== 'lobby' && (
        <p className="policy-help">Leave the race to change policy.</p>
      )}
    </section>
  );
}
