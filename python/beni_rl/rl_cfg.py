# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""PPO configuration for BENI training."""

from dataclasses import asdict, replace
from pathlib import Path
import math
import json
import shutil
import torch
import yaml
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.rl.runner import MjlabOnPolicyRunner

from .recovery_gate import RECOVERY_PROTOCOL
from .recovery_bank import bank_digest, update_hard_states
from .jump_evaluation import JUMP_EVALUATION_PROTOCOL
from .recovery_curriculum import grounding_probability, probabilities, consolidation_progress


def recovery_gate_signature(cfg):
    """Compare settings independently of the checkout's absolute asset location."""
    settings = dict(cfg) if isinstance(cfg,dict) else asdict(cfg)
    settings.pop('heldout_bank',None)
    return settings


@torch.inference_mode()
def prime_jump_normalizers(actor, critic, observation_manager, allow_existing_prior=False):
    """Initialize unit-scale normalization for Jump commands at the curriculum transition."""
    record = {}
    for name, model in (('actor', actor), ('critic', critic)):
        names = observation_manager.active_terms[name]
        dims = observation_manager.group_obs_term_dim[name]
        index = names.index('jump_command')
        start = sum(math.prod(d) for d in dims[:index])
        if dims[index] != (4,):
            raise ValueError('Expected four Jump observation fields')
        sl = slice(start, start+4)
        norm = model.obs_normalizer
        if torch.count_nonzero(norm._mean[..., sl]) or (
                not allow_existing_prior and torch.count_nonzero(norm._var[..., sl])):
            raise ValueError('Jump normalization prior requires the unused zero-command prefix')
        record[name] = dict(columns=list(range(start,start+4)), count=int(norm.count),
                            before_std=norm._std[..., sl].flatten().tolist(), after_std=1.-norm.eps)
        norm._var[..., sl] = (1.-norm.eps)**2
        norm._std[..., sl] = 1.-norm.eps
    return record


class BeniVelocityOnPolicyRunner(MjlabOnPolicyRunner):
    """Train the automatic curriculum and export recurrent policies with checkpoints."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        if log_dir is not None:
            env.unwrapped._diagnostics_dir = str(Path(log_dir) / 'diagnostics' / 'wheel_overspeed')
        # mjlab offsets the worker seed by rank; curriculum checks use the common base seed.
        self._curriculum_seed = self.cfg['seed'] - self.gpu_global_rank
        if self.is_distributed:
            from .distributed import synchronize_normalizers
            synchronize_normalizers(self.alg.actor, self.alg.critic)
        self._completed_updates = 0
        self._jump_state = None
        self._jump_prior_primed = False
        self._last_recovery_check = 0
        self._recovery_passes = 0
        self._supine_grounding_probability = 0.
        self._last_supine_check = 0
        self._supine_sampling_restore_start = None
        self._consolidation = None
        self._hard_failures = {}
        self._hard_groups = {}
        bank_event = env.unwrapped.cfg.events.get('reset_curriculum')
        self._bank_sha256 = bank_digest(bank_event.params['bank_path']) if bank_event is not None else None
        self._resumed = False
        self._best_checkpoint = None
        self._last_best_evaluation_updates = None
        self._last_performance_check = 0
        self._best_recovery_checkpoint = None
        update = self.alg.update
        def update_curriculum(*args, **kwargs):
            losses = update(*args, **kwargs)
            self._completed_updates += 1
            self._update_recovery_sampling()
            self._check_recovery()
            self._check_performance()
            return losses
        self.alg.update = update_curriculum
        if "jump" in env.unwrapped.cfg.commands:
            self._configure_supine_curriculum()
            # Collect minibatch KL values for logging.
            kl_values = []
            original_kl = self.alg.actor.get_kl_divergence
            original_log = self.logger.log
            def record_kl(*args, **kwargs):
                value = original_kl(*args, **kwargs)
                kl_values.append(value.detach().mean())
                return value
            def log_jump(**kwargs):
                losses = dict(kwargs["loss_dict"])
                if kl_values:
                    losses["kl"] = float(torch.stack(kl_values).mean())
                    kl_values.clear()
                if not all(math.isfinite(float(v)) for v in losses.values()) or not torch.isfinite(kwargs["action_std"]).all():
                    failed = Path(self.logger.log_dir) / f"failed_nonfinite_{kwargs['it']}.pt"
                    if self.gpu_global_rank == 0:
                        MjlabOnPolicyRunner.save(self, str(failed))
                    raise FloatingPointError(f"Nonfinite optimizer metrics; failed state saved to {failed}: {losses}")
                kwargs["loss_dict"] = losses
                original_log(**kwargs)
                if self.logger.writer is not None:
                    self.logger.writer.add_scalar('Curriculum/jump_enabled', int(self._jump_state is not None), kwargs['it'])
            self.alg.actor.get_kl_divergence = record_kl
            self.logger.log = log_jump

    def _configure_jump(self):
        env = self.env.unwrapped
        command = env.command_manager.get_term('jump')
        state = self._jump_state or {}
        command.configure_training(self._completed_updates, self.cfg['num_steps_per_env'],
                                   jump_start_iteration=state.get('start_iteration'),
                                   ramp_iterations=state.get('ramp_iterations'))
        self._configure_supine_curriculum()

    def _configure_supine_curriculum(self):
        env = self.env.unwrapped
        cfg = env.cfg.commands['jump'].recovery_gate
        if self._completed_updates >= cfg.supine_start_iteration and self._supine_grounding_probability == 0:
            # Keep some grounded training data before requiring success on that source.
            self._supine_grounding_probability = cfg.supine_initial_grounding_probability
        env._supine_grounding_probability = self._supine_grounding_probability
        if (self._jump_state is not None and self._consolidation is None
                and self._supine_grounding_probability >= 1 and self._supine_sampling_restore_start is None):
            self._supine_sampling_restore_start = self._completed_updates
        env._supine_sampling_restore_start = self._supine_sampling_restore_start
        if (self._jump_state is None and self._consolidation is None
                and self._supine_grounding_probability >= 1 and grounding_probability(self._completed_updates) >= 1):
            self._consolidation = dict(start_updates=self._completed_updates,
                                       initial_probabilities=probabilities(self._completed_updates,
                                                                          self._supine_sampling_restore_start))
            self._recovery_passes = 0
        env._recovery_consolidation = self._consolidation
        sampler = getattr(env, '_recovery_sampler', None)
        if self._consolidation is not None and sampler is not None:
            sampler.start(self._completed_updates)

    def _configure_hard_sampling(self):
        self.env.unwrapped._hard_recovery_groups = {
            kind: [torch.tensor(indices, device=self.device, dtype=torch.long)
                   for _, indices in sorted(groups.items()) if indices]
            for kind, groups in self._hard_groups.items()}

    def _update_recovery_sampling(self):
        sampler = getattr(self.env.unwrapped, '_recovery_sampler', None)
        if sampler is None or not sampler.update(self._completed_updates):
            return
        if self.gpu_global_rank == 0:
            record = dict(completed_updates=self._completed_updates, **sampler.last_metrics)
            if self.logger.log_dir:
                directory = Path(self.logger.log_dir) / 'curriculum' / 'sampling'
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f'{self._completed_updates}.json').write_text(json.dumps(record, indent=2)+'\n')
            if self.logger.writer is not None:
                for name, value in sampler.last_metrics.items():
                    self.logger.writer.add_scalar('RecoverySampling/'+name, value, self._completed_updates-1)

    def _evaluate_recovery(self, cfg):
        from .recovery_gate import check_recovery
        result = None
        if self.gpu_global_rank == 0:
            try:
                result = check_recovery(self.env.unwrapped.cfg, self.alg.actor, cfg, self._curriculum_seed, self.device)
            except Exception as exc:
                result = {'error': f'{type(exc).__name__}: {exc}'}
        if self.is_distributed:
            values = [result]
            torch.distributed.broadcast_object_list(values, src=0, device=torch.device(self.device))
            result = values[0]
        if 'error' in result:
            raise RuntimeError(f'Recovery curriculum check failed: {result["error"]}')
        return result

    def _check_jump_deadline(self):
        cfg = self.env.unwrapped.cfg.commands['jump'].recovery_gate
        if (self._jump_state is None and cfg.force_jump_after_updates is not None
                and self._completed_updates >= cfg.force_jump_after_updates):
            self._enter_jump('update_limit')
            return True
        return False

    def _check_recovery(self):
        env = self.env.unwrapped
        if 'jump' not in env.cfg.commands:
            return
        command = env.command_manager.get_term('jump')
        cfg = command.cfg.recovery_gate
        self._configure_supine_curriculum()
        # The update limit is independent of assessment timing and attained difficulty.
        if self._check_jump_deadline():
            return
        full_due = (self._completed_updates >= command.cfg.warmup_iterations
                    and self._completed_updates-self._last_recovery_check >= cfg.check_interval)
        supine_due = (self._supine_grounding_probability < 1
                      and self._completed_updates >= cfg.supine_start_iteration+cfg.check_interval
                      and self._completed_updates-self._last_supine_check >= cfg.check_interval)
        if not full_due and not supine_due:
            return
        result = self._evaluate_recovery(cfg) if full_due else None
        if supine_due:
            # Reuse the full check when it includes supine; never let other classes gate this increment.
            measured = result if result is not None and 'supine' in result['scenarios'] else self._evaluate_recovery(
                replace(cfg, scenarios=('supine',), subgroups=()))
            row = measured['scenarios']['supine']
            before = self._supine_grounding_probability
            supine_passed = row['valid'] >= 32 and row['recovered'] >= cfg.supine_success_rate*row['valid']
            if supine_passed:
                self._supine_grounding_probability = min(1., before+cfg.supine_grounding_step)
            self._last_supine_check = self._completed_updates
            self._configure_supine_curriculum()
            record = dict(completed_updates=self._completed_updates, before=before,
                          after=self._supine_grounding_probability, assessment=row,
                          advancement_passed=supine_passed, advancement_success_rate=cfg.supine_success_rate,
                          protocol=measured.get('protocol'), seed=self._curriculum_seed,
                          bank_path=measured.get('bank_path'), choices=measured.get('choices', {}).get('supine'))
            if self.gpu_global_rank == 0:
                if self.logger.log_dir:
                    directory = Path(self.logger.log_dir)/'curriculum'
                    directory.mkdir(exist_ok=True, parents=True)
                    (directory/f'supine_{self._completed_updates}.json').write_text(json.dumps(record, indent=2)+'\n')
                if self.logger.writer is not None:
                    self.logger.writer.add_scalar('SupineCurriculum/grounded_probability', record['after'],
                                                  self._completed_updates-1)
                    for key in ('valid', 'invalid', 'recovered', 'success_rate', 'passed'):
                        value = supine_passed if key == 'passed' else row[key]
                        if value is not None:
                            self.logger.writer.add_scalar(f'SupineCurriculum/{key}', value, self._completed_updates-1)
                print('[Supine curriculum]', json.dumps(record), flush=True)
        if not full_due:
            return
        self._last_recovery_check = self._completed_updates
        if self._consolidation is None:
            update_hard_states(self._hard_failures, self._hard_groups, result)
            self._configure_hard_sampling()
        target_reached = consolidation_progress(self._completed_updates, self._consolidation) >= 1
        self._recovery_passes = self._recovery_passes+1 if target_reached and result['passed'] else 0
        result.update(completed_updates=self._completed_updates, consecutive_passes=self._recovery_passes)
        if self.gpu_global_rank == 0:
            if self.logger.log_dir:
                output = Path(self.logger.log_dir)/'curriculum'
                output.mkdir(exist_ok=True, parents=True)
                (output/f'recovery_{self._completed_updates}.json').write_text(json.dumps(result, indent=2)+'\n')
            if self.logger.writer is not None:
                self.logger.writer.add_scalar('RecoveryGate/consecutive_passes', self._recovery_passes,
                                              self._completed_updates-1)
                for name, row in result['scenarios'].items():
                    for key in ('valid', 'invalid', 'recovered', 'success_rate', 'passed'):
                        if row[key] is not None:
                            self.logger.writer.add_scalar(f'RecoveryGate/{name}/{key}', row[key], self._completed_updates-1)
            print('[Recovery curriculum]', self._completed_updates, result['scenarios'], flush=True)
        if self._jump_state is not None or self._recovery_passes < cfg.consecutive_passes:
            return
        self._enter_jump('recovery_gate')

    def _enter_jump(self, reason):
        """Enable the Jump probability ramp while all PPO parameters keep learning."""
        if self._jump_state is not None:
            return
        env = self.env.unwrapped
        command = env.command_manager.get_term('jump')
        prior = prime_jump_normalizers(self.alg.actor, self.alg.critic, env.observation_manager,
                                       allow_existing_prior=self._jump_prior_primed)
        self._jump_prior_primed = True
        self._jump_state = dict(start_iteration=self._completed_updates, ramp_iterations=command.cfg.ramp_iterations,
                                learning_mode='full_actor', transition_reason=reason,
                                recovery_passes=self._recovery_passes,
                                supine_grounding_probability=self._supine_grounding_probability)
        self._configure_jump()
        if self.gpu_global_rank == 0:
            if self.logger.log_dir:
                root = Path(self.logger.log_dir)
                (root/'jump_normalizer_prior.json').write_text(json.dumps(prior, indent=2)+'\n')
                (root/'jump_transition.json').write_text(json.dumps(self._jump_state, indent=2)+'\n')
                previous_label = self.current_learning_iteration
                try:
                    self.current_learning_iteration = self._completed_updates-1
                    self.save(str(root/f'model_{self.current_learning_iteration}.pt'))
                finally:
                    self.current_learning_iteration = previous_label
            print('[Jump curriculum enabled]', json.dumps(self._jump_state), flush=True)

    def _check_performance(self):
        env = self.env.unwrapped
        if 'jump' not in env.cfg.commands:
            return
        cfg = env.cfg.commands['jump'].recovery_gate
        if (self._completed_updates < env.cfg.commands['jump'].warmup_iterations
                or self._completed_updates-self._last_performance_check < cfg.performance_interval):
            return
        result = None
        if self.gpu_global_rank == 0:
            try:
                from .recovery_gate import check_recovery
                from .recovery_evaluation import evaluate_locomotion
                result = dict(completed_updates=self._completed_updates,
                              recovery=check_recovery(env.cfg, self.alg.actor, cfg, cfg.heldout_seed,
                                                      self.device, bank_path=cfg.heldout_bank),
                              locomotion=evaluate_locomotion(env.cfg, self.alg.actor, cfg.heldout_seed, self.device))
                if self._jump_state is not None:
                    from .jump_evaluation import evaluate_jump_checkpoint
                    result['jump'] = evaluate_jump_checkpoint(env.cfg, self.alg.actor,
                                                             env.cfg.commands['jump'].best_checkpoint, self.device)
            except Exception as exc:
                result = dict(completed_updates=self._completed_updates, error=f'{type(exc).__name__}: {exc}')
        if self.is_distributed:
            values = [result]
            torch.distributed.broadcast_object_list(values, src=0, device=torch.device(self.device))
            result = values[0]
        self._last_performance_check = self._completed_updates
        if self.gpu_global_rank != 0:
            return
        root = Path(self.logger.log_dir) if self.logger.log_dir else None
        if root is not None:
            result['checkpoint'] = f'model_recovery_{self._completed_updates-1}.pt'
            directory = root/'performance'; directory.mkdir(parents=True, exist_ok=True)
            (directory/f'{self._completed_updates}.json').write_text(json.dumps(result, indent=2)+'\n')
        if 'error' in result:
            print('[Performance assessment failed]', result['error'], flush=True)
            return
        rows = result['recovery']['scenarios']
        covered = all(row['valid'] >= cfg.min_valid_trials for row in rows.values())
        worst = min((row['success_rate'] or 0.) for row in rows.values())
        if self.logger.writer is not None:
            self.logger.writer.add_scalar('HeldoutRecovery/worst_subgroup_success_rate', worst, self._completed_updates-1)
            self.logger.writer.add_scalar('HeldoutRecovery/covered', covered, self._completed_updates-1)
            self.logger.writer.add_scalar('HeldoutLocomotion/passed', result['locomotion']['passed'], self._completed_updates-1)
        print('[Independent performance]', self._completed_updates, 'worst recovery:', worst,
              'coverage:', covered, 'locomotion:', result['locomotion']['passed'], flush=True)
        if self._jump_state is not None and (not result['recovery']['passed'] or not result['locomotion']['passed']):
            print('[Recovery after Jump] Assessment below threshold; inspect performance JSON. '
                  'Actor and critic continue joint training.', flush=True)
        score = [worst, -max((row['recovery_time_p90_s'] if row['recovery_time_p90_s'] is not None
                              else cfg.deadline_s+cfg.hold_s) for row in rows.values())]
        improved = (root is not None and covered and result['locomotion']['passed']
                    and (self._best_recovery_checkpoint is None or score > self._best_recovery_checkpoint['score']))
        if improved:
            record = dict(score=score, completed_updates=self._completed_updates,
                          checkpoint=f'model_recovery_{self._completed_updates-1}.pt',
                          protocol=RECOVERY_PROTOCOL, bank_sha256=result['recovery']['bank_sha256'],
                          seed=cfg.heldout_seed, evaluation=asdict(cfg))
            self._best_recovery_checkpoint = record
        if root is not None:
            previous = self.current_learning_iteration
            try:
                self.current_learning_iteration = self._completed_updates-1
                self.save(str(root/result['checkpoint']), evaluate_jump=False)
            finally:
                self.current_learning_iteration = previous
            if improved:
                self._write_best_checkpoint(root/result['checkpoint'], record, stem='model_recovery_best')

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        env = self.env.unwrapped
        # On resume, num_learning_iterations specifies additional updates.
        self.current_learning_iteration = self._completed_updates
        if "jump" in env.cfg.commands:
            self._configure_jump()
            self._configure_supine_curriculum()
            # A resumed checkpoint may already exceed the configured Jump update limit.
            self._check_jump_deadline()
        if self._resumed:
            # Checkpoints contain model/optimizer/curriculum state, not simulator episodes.
            self.env.reset()
            self.alg.actor.reset()
            self.alg.critic.reset()
            self._resumed = False
        if "fallen_reset_probability" in env.curriculum_manager.active_terms:
            curriculum = env.curriculum_manager.get_term_cfg("fallen_reset_probability")
            curriculum.params.update(
                steps_per_iteration=self.cfg["num_steps_per_env"],
                start_iteration=self.current_learning_iteration,
                start_step=env.common_step_counter,
            )
            # Begin the configured reset curriculum with fresh episodes and GRU state.
            self.env.reset()
            self.alg.actor.reset()
            self.alg.critic.reset()
        try:
            super().learn(num_learning_iterations, init_at_random_ep_len)
        except Exception:
            recorder = getattr(env, '_wheel_overspeed', None)
            if recorder is not None:
                try:
                    recorder.dump_failure(env)
                except Exception as exc:
                    print(f'[WARN] Failed to save wheel diagnostics: {exc}', flush=True)
            raise
        if 'jump' in env.cfg.commands and self._jump_state is None:
            print('[Recovery curriculum] Budget finished in foundation; recovery has not passed and Jump remains disabled.',
                  flush=True)
        if 'jump' in env.cfg.commands and env.cfg.commands['jump'].best_checkpoint.enabled and self.logger.log_dir:
            if self._best_checkpoint is None:
                print('[Best Jump checkpoint] No eligible checkpoints were evaluated.', flush=True)
            else:
                print('[Best Jump checkpoint]', Path(self.logger.log_dir)/'model_best.pt',
                      'completed updates:', self._best_checkpoint['completed_updates'], flush=True)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        # Clone inference-mode buffers before loading checkpoint values.
        with torch.inference_mode(False):
            for model in (self.alg.actor, self.alg.critic):
                for module in model.modules():
                    for name, value in module.named_buffers(recurse=False):
                        if value.is_inference():
                            module._buffers[name] = value.clone()
        if load_cfg is not None and not load_cfg.get('iteration', False):
            return super().load(path, load_cfg, strict, map_location or self.device)
        infos = super().load(path, load_cfg, strict, map_location or self.device) or {}
        state = infos.get('beni_training', {})
        self._completed_updates = state.get('completed_updates', self.current_learning_iteration+1)
        # Continue at the update following the checkpoint's last completed label.
        self.current_learning_iteration = self._completed_updates
        self.env.unwrapped.common_step_counter = self._completed_updates*self.cfg['num_steps_per_env']
        self._resumed = True
        self._last_recovery_check = state.get('last_recovery_check', 0)
        self._recovery_passes = state.get('recovery_passes', 0)
        self._last_performance_check = state.get('last_performance_check', 0)
        self._best_recovery_checkpoint = state.get('best_recovery_checkpoint')
        self._hard_failures = state.get('hard_failures', {})
        self._hard_groups = state.get('hard_groups', {})
        if state.get('bank_sha256') != self._bank_sha256:
            self._hard_failures = {}; self._hard_groups = {}
            self._recovery_passes = 0
            self._last_recovery_check = 0
        self._configure_hard_sampling()
        sampler = getattr(self.env.unwrapped, '_recovery_sampler', None)
        if sampler is not None:
            if not sampler.load_state_dict(state.get('recovery_sampling')):
                # Incompatible sampling history requires fresh recovery gate evidence.
                self._recovery_passes = 0
                sampler.last_update = self._completed_updates
            if sampler.last_update > self._completed_updates:
                raise ValueError('Recovery sampling state is ahead of the saved training iteration')
        if 'jump' not in self.env.unwrapped.cfg.commands:
            return infos
        # Checkpoints without a saved supine level use their elapsed update count.
        self._supine_grounding_probability = state.get('supine_grounding_probability',
                                                       grounding_probability(self._completed_updates))
        if not 0 <= self._supine_grounding_probability <= 1:
            raise ValueError('Invalid saved supine grounding probability')
        self._last_supine_check = state.get('last_supine_check', self._completed_updates)
        self._supine_sampling_restore_start = state.get('supine_sampling_restore_start')
        self._consolidation = state.get('consolidation')
        if self._consolidation is not None:
            initial = self._consolidation['initial_probabilities']
            if (len(initial) != 9 or any(not math.isfinite(p) or p < 0 for p in initial)
                    or not math.isclose(sum(initial), 1., abs_tol=1e-6)
                    or not 0 <= self._consolidation['start_updates'] <= self._completed_updates):
                raise ValueError('Invalid saved recovery consolidation schedule')
        self._jump_state = state.get('jump')
        self._jump_prior_primed = state.get('jump_prior_primed', False)
        gate = self.env.unwrapped.cfg.commands['jump'].recovery_gate
        if state and (recovery_gate_signature(state.get('recovery_gate', {})) != recovery_gate_signature(gate) or state.get('recovery_seed') != self._curriculum_seed
                      or state.get('recovery_protocol') != RECOVERY_PROTOCOL):
            self._recovery_passes = 0
            self._last_recovery_check = 0
        if not state:
            self._load_legacy_curriculum(Path(path))
        self._configure_supine_curriculum()
        if self._jump_state is not None:
            # Preserve Jump timing while discarding obsolete actor freezing state.
            self._jump_state = {key: value for key, value in self._jump_state.items() if key != 'freeze'}
            self._jump_state['learning_mode'] = 'full_actor'
            if self.gpu_global_rank == 0 and self.logger.log_dir:
                root = Path(self.logger.log_dir)
                root.mkdir(parents=True, exist_ok=True)
                (root/'jump_transition.json').write_text(json.dumps(self._jump_state, indent=2)+'\n')
        self._restore_best_checkpoint(path, state.get('best_checkpoint'))
        self._restore_best_recovery(path, state.get('best_recovery_checkpoint'))
        return infos

    def _restore_best_recovery(self, path, record):
        self._best_recovery_checkpoint = None
        cfg = self.env.unwrapped.cfg.commands['jump'].recovery_gate
        if (not record or record.get('protocol') != RECOVERY_PROTOCOL
                or recovery_gate_signature(record.get('evaluation', {})) != recovery_gate_signature(cfg)
                or not Path(cfg.heldout_bank).is_file()
                or record.get('bank_sha256') != bank_digest(cfg.heldout_bank)
                or not record.get('score') or not all(map(math.isfinite,record['score']))):
            return
        source = (Path(path) if record['completed_updates'] == self._completed_updates
                  else Path(path).parent/Path(record['checkpoint']).name)
        if not source.is_file():
            candidate = Path(path).parent/'model_recovery_best.pt'
            if candidate.is_file():
                saved = torch.load(candidate,map_location='cpu',weights_only=True)['infos']['beni_training']
                if (saved['completed_updates'] == record['completed_updates']
                        and saved.get('best_recovery_checkpoint') == record):
                    source = candidate
        if not source.is_file():
            print('[Best recovery checkpoint] Earlier selected policy is unavailable; selection restarts.',flush=True)
            return
        self._best_recovery_checkpoint = record
        if self.gpu_global_rank == 0 and self.logger.log_dir:
            root = Path(self.logger.log_dir); root.mkdir(parents=True,exist_ok=True)
            self._write_best_checkpoint(source, record, stem='model_recovery_best')

    def _write_best_checkpoint(self, source, record, stem='model_best'):
        root = Path(self.logger.log_dir)
        root.mkdir(parents=True, exist_ok=True)
        destination = root/f'{stem}.pt'
        if Path(source).resolve() != destination.resolve():
            temporary = root/f'.{stem}.pt.tmp'
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        temporary = root/f'.{stem}.json.tmp'
        temporary.write_text(json.dumps(record, indent=2)+'\n')
        temporary.replace(root/f'{stem}.json')

    def _restore_best_checkpoint(self, path, record):
        self._best_checkpoint = None
        self._last_best_evaluation_updates = None
        cfg = self.env.unwrapped.cfg.commands['jump'].best_checkpoint
        if (not cfg.enabled or not record or record.get('evaluation') != asdict(cfg)
                or record.get('protocol') != JUMP_EVALUATION_PROTOCOL
                or not record.get('best_eligible')
                or not math.isfinite(record.get('median_clearance_m', float('nan')))
                or record['median_clearance_m'] <= cfg.min_median_clearance_m
                or not record.get('score') or not all(map(math.isfinite, record['score']))):
            return
        source = (Path(path) if record['completed_updates'] == self._completed_updates
                  else Path(path).parent/Path(record['checkpoint']).name)
        if not source.is_file():
            candidate = Path(path).parent/'model_best.pt'
            if candidate.is_file():
                try:
                    saved = torch.load(candidate, map_location='cpu', weights_only=True)['infos']['beni_training']
                    if (saved['completed_updates'] == record['completed_updates']
                            and saved['best_checkpoint']['score'] == record['score']
                            and saved['best_checkpoint']['evaluation'] == record['evaluation']):
                        source = candidate
                except Exception as exc:
                    print(f'[WARN] Earlier best checkpoint could not be restored: {exc}', flush=True)
        if not source.is_file():
            print('[Best Jump checkpoint] Earlier selected checkpoint is unavailable; selection restarts.', flush=True)
            return
        self._best_checkpoint = record
        if self.gpu_global_rank == 0 and self.logger.log_dir:
            self._write_best_checkpoint(source, record)

    def _evaluate_best_checkpoint(self, path):
        env = self.env.unwrapped
        if 'jump' not in env.cfg.commands or not self.logger.log_dir or self.gpu_global_rank != 0:
            return False
        cfg = env.cfg.commands['jump'].best_checkpoint
        if (not cfg.enabled or self._jump_state is None or self._completed_updates < cfg.start_updates
                or self._completed_updates <= self._jump_state['start_iteration']
                or self._last_best_evaluation_updates == self._completed_updates):
            return False
        from .jump_evaluation import evaluate_jump_checkpoint
        try:
            result = evaluate_jump_checkpoint(env.cfg, self.alg.actor, cfg, self.device)
            if not result['score'] or not all(map(math.isfinite, result['score'])):
                raise ValueError('Nonfinite or empty Jump checkpoint score')
            record = dict(checkpoint=Path(path).name, completed_updates=self._completed_updates,
                          evaluation=asdict(cfg), **result)
            directory = Path(self.logger.log_dir)/'checkpoint_evaluations'
            directory.mkdir(parents=True, exist_ok=True)
            (directory/(Path(path).stem+'.json')).write_text(json.dumps(record, indent=2)+'\n')
            if self.logger.writer is not None:
                for key in ('heading_directional_landing_rate', 'heading_pass_rate',
                            'mean_peak_heading_error_deg', 'mean_final_heading_error_deg',
                            'directional_landing_rate', 'landing_rate', 'takeoff_rate',
                            'strict_success_rate', 'mean_joint_limit_violation_rad',
                            'median_clearance_m', 'best_eligible'):
                    self.logger.writer.add_scalar('JumpEvaluation/'+key, record[key], self.current_learning_iteration)
            improved = record['best_eligible'] and (
                self._best_checkpoint is None or tuple(record['score']) > tuple(self._best_checkpoint['score']))
            if improved:
                self._best_checkpoint = record
            self._last_best_evaluation_updates = self._completed_updates
            print('[Jump checkpoint evaluation]', Path(path).name, 'score:', record['score'],
                  'median clearance:', record['median_clearance_m'],
                  'eligible:', record['best_eligible'], 'best:', improved, flush=True)
            return improved
        except Exception as exc:
            print(f'[WARN] Jump checkpoint evaluation failed (training continues): {exc}', flush=True)
            return False

    def _load_legacy_curriculum(self, path):
        """Recover older checkpoints' stage from their original run metadata."""
        config_path = path.parent/'params/env.yaml'
        if not config_path.is_file():
            raise ValueError('Older Jump checkpoints require their original params/env.yaml to recover the curriculum')
        config = yaml.load(config_path.read_text(), Loader=yaml.BaseLoader)['commands']['jump']
        start = int(config['warmup_iterations'])
        norm = self.alg.actor.obs_normalizer
        self._jump_prior_primed = bool(torch.count_nonzero(norm._var[..., 36:40]))
        if self._completed_updates <= start:
            return
        freeze_path = path.parent/'jump_actor_freeze.json'
        if not freeze_path.is_file():
            raise ValueError('Older Jump-stage checkpoints require jump_actor_freeze.json to recover their original Jump start')
        frozen = json.loads(freeze_path.read_text())
        start = int(frozen['completed_foundation_updates'])
        ramp = min(1000., max(1., (int(config['total_iterations'])-start)/3))
        self._jump_state = dict(start_iteration=start, ramp_iterations=ramp, learning_mode='full_actor')

    def save(self, path: str, infos=None, *, evaluate_jump=True) -> None:
        if self.gpu_global_rank != 0:
            return
        improved = self._evaluate_best_checkpoint(path) if evaluate_jump else False
        infos = {**(infos or {}), 'beni_training': dict(
            completed_updates=self._completed_updates, jump=self._jump_state,
            jump_prior_primed=self._jump_prior_primed, last_recovery_check=self._last_recovery_check,
            recovery_passes=self._recovery_passes, best_checkpoint=self._best_checkpoint,
            bank_sha256=self._bank_sha256, hard_failures=self._hard_failures, hard_groups=self._hard_groups,
            last_performance_check=self._last_performance_check, best_recovery_checkpoint=self._best_recovery_checkpoint)}
        sampler = getattr(self.env.unwrapped, '_recovery_sampler', None)
        if sampler is not None:
            infos['beni_training']['recovery_sampling'] = sampler.state_dict()
        if 'jump' in self.env.unwrapped.cfg.commands:
            infos['beni_training'].update(recovery_gate=asdict(self.env.unwrapped.cfg.commands['jump'].recovery_gate),
                                          recovery_seed=self._curriculum_seed, recovery_protocol=RECOVERY_PROTOCOL,
                                          supine_grounding_probability=self._supine_grounding_probability,
                                          last_supine_check=self._last_supine_check,
                                          supine_sampling_restore_start=self._supine_sampling_restore_start,
                                          consolidation=self._consolidation)
        super().save(path, infos)
        if improved:
            self._write_best_checkpoint(path, self._best_checkpoint)
        policy_dir, filename, _ = self._get_export_paths(path)
        filename = Path(path).stem + ".onnx"
        try:
            self.export_policy_to_onnx(str(policy_dir), filename)
        except Exception as exc:
            print(f"[WARN] ONNX export failed (training continues): {exc}")


def beni_velocity_flat_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    common_model = {
        "hidden_dims": (512, 256, 128),
        "activation": "elu",
        "obs_normalization": True,
        "rnn_type": "gru",
        "rnn_hidden_dim": 512,
        "rnn_num_layers": 1,
        "class_name": "RNNModel",
    }
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            **common_model,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(**common_model),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="beni_velocity_flat",
        logger="tensorboard",
        upload_model=False,
        seed=42,
        obs_groups={"actor": ("actor",), "critic": ("critic",)},
        clip_actions=None,
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=10000,
    )


def beni_jump_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return replace(beni_velocity_flat_ppo_runner_cfg(), experiment_name="beni_jump_flat",
                   max_iterations=8501, save_interval=500, load_checkpoint=r"model_\d+\.pt")
