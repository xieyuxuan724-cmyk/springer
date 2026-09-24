import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ArrowRight } from 'lucide-react';
import { useNameForm } from './context';
export function LoginPanel() {
  const { draftName, setDraftName, submitName } = useNameForm();
  return (
    <div className="start-panel">
      <h1>Ready to race?</h1>
      <form onSubmit={submitName}>
        <label htmlFor="name">Driver name</label>
        <Input
          id="name"
          autoComplete="nickname"
          placeholder="Choose a name"
          value={draftName}
          onChange={(e) => setDraftName(e.target.value)}
          maxLength={40}
          required
        />
        <Button className="primary" type="submit">
          Enter garage <ArrowRight size={18} />
        </Button>
      </form>
    </div>
  );
}
