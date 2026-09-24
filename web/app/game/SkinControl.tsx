import { useGame } from './context';
import { SKINS, getSkin } from '../../game/skins.mjs';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { ChevronDown } from 'lucide-react';
function SkinPicker({
  value,
  taken,
  disabled,
  onPick,
}: {
  value: string;
  taken: string[];
  disabled?: boolean;
  onPick: (id: string) => void;
}) {
  return (
    <RadioGroup
      value={value}
      onValueChange={(v) => onPick(String(v))}
      className="skin-picker"
      aria-label="Robot skins"
    >
      {SKINS.map((s) => {
        const used = s.id !== value && taken.includes(s.id);
        return (
          <label
            key={s.id}
            className={`skin-chip ${s.id === value ? 'selected' : ''} ${
              used ? 'taken' : ''
            }`}
            style={{ '--skin-color': s.primary } as React.CSSProperties}
            title={used ? `${s.name} (taken)` : s.name}
          >
            <RadioGroupItem
              value={s.id}
              className="sr-only"
              disabled={disabled || used}
            />
            <span className="skin-dot" />
            {s.name}
          </label>
        );
      })}
    </RadioGroup>
  );
}

export function SkinControl() {
  const { skinId, takenSkins, connected, send } = useGame();
  return (
    <details className="skin-control">
      <summary>
        <span>Skin</span>
        <span className="skin-current">
          <span
            className="skin-dot"
            style={
              { '--skin-color': getSkin(skinId).primary } as React.CSSProperties
            }
          />
          {getSkin(skinId).name}
        </span>
        <ChevronDown size={15} />
      </summary>
      <SkinPicker
        value={skinId}
        taken={takenSkins}
        disabled={!connected}
        onPick={(id) => send({ type: 'skin', skin: id })}
      />
    </details>
  );
}
