# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Recovery state sampling and training curriculum schedules."""
import math
from pathlib import Path

import torch
from mjlab.envs import mdp as envs_mdp

from .recovery import restore_recorded_state
from . import ASSETS_DIR

KINDS = ('normal', 'supine', 'prone', 'upside_down', 'belly_support',
         'leg_support', 'head_support', 'one_wheel', 'mixed_support')
BANK_VERSION = 2
CONSOLIDATION_UPDATES = 500
DEFAULT_TRAINING_STATE_BANK = ASSETS_DIR / 'motions' / 'recovery_contact_states_train.npz'


def consolidation_progress(iteration, state):
    return 0. if state is None else min(1., max(0., (iteration-state['start_updates'])/CONSOLIDATION_UPDATES))



def training_state_bank(path=None):
    """Return a training bank after checking its recovery-state schema."""
    path = Path(path) if path is not None else DEFAULT_TRAINING_STATE_BANK
    from .recovery_bank import load_recovery_bank
    load_recovery_bank(path)
    return path


def probabilities(iteration, supine_sampling_restore_start=None, consolidation=None, recovery_probabilities=None):
    """Ramp to 25% supine resets between updates 750 and 3000.

    After all grounding curricula finish, blend to 20% normal over 500 updates.
    After supine grounding is mastered, gradually reduce supine reset sampling.
    """
    if consolidation is not None:
        if recovery_probabilities is None:
            raise ValueError('Consolidation requires the actual whole-bank recovery probabilities')
        fraction = consolidation_progress(iteration, consolidation)
        target = [.20, *(.8 * p for p in recovery_probabilities)]
        return [(1-fraction)*initial + fraction*target for initial, target in
                zip(consolidation['initial_probabilities'], target)]
    progress = min(1., max(0., (iteration - 750) / 2250))
    restored = (0. if supine_sampling_restore_start is None else
                min(1., max(0., (iteration - supine_sampling_restore_start) / 500)))
    supine = .25 + (.40 / 3 - .25) * restored
    recovery = [supine * progress, *([.40 / 3 * progress] * 2), *([.03 * progress] * 5)]
    return [1 - sum(recovery), *recovery]


def grounding_probability(iteration):
    """Ramp grounded-fall probability from zero at update 1500 to one at 3000."""
    return min(1., max(0., (iteration - 1500) / 1500))


def reset_airborne_fall(env, ids, kind):
    """Reset the root to a perturbed fall pose 0.06 m above its default height."""
    from mjlab.utils.lab_api.math import quat_from_euler_xyz
    robot = env.scene['robot']
    state = robot.data.default_root_state[ids].clone()
    state[:, :3] += env.scene.env_origins[ids]
    state[:, 2] += .06
    angles = torch.empty(len(ids), 3, device=env.device).uniform_(-.15, .15)
    angles[:, 2].uniform_(-math.pi, math.pi)
    if kind == 3:
        angles[:, 0] += math.pi
    else:
        angles[:, 1] += -math.pi / 2 if kind == 1 else math.pi / 2
    state[:, 3:7] = quat_from_euler_xyz(*angles.unbind(-1))
    robot.write_root_link_pose_to_sim(state[:, :7], env_ids=ids)
    robot.write_root_link_velocity_to_sim(torch.zeros_like(state[:, 7:]), env_ids=ids)
    # Joint positions and velocities remain as sampled by the joint reset term.


def reset_curriculum_states(env, env_ids, bank_path, steps_per_iteration=24):
    ids = envs_mdp.resolve_env_ids(env, env_ids)
    sampler = env._recovery_sampler
    if not hasattr(env, '_curriculum_kind'):
        env._curriculum_kind = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        env._curriculum_counts = torch.zeros(len(KINDS), device=env.device)
    if not hasattr(env, '_fall_source_counts'):
        env._fall_source_counts = torch.zeros(2, device=env.device)
        env._curriculum_airborne = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._curriculum_airborne[ids] = False
    env._recovery_state_id[ids] = -1
    iteration = env.common_step_counter // steps_per_iteration
    grounded_probability = grounding_probability(iteration)
    supine_probability = getattr(env, '_supine_grounding_probability', grounded_probability)
    consolidation = getattr(env, '_recovery_consolidation', None)
    fraction = consolidation_progress(iteration, consolidation)
    initial = (consolidation['initial_probabilities'] if consolidation is not None else
               probabilities(iteration, getattr(env, '_supine_sampling_restore_start', None)))
    kinds = torch.multinomial(torch.tensor(initial, device=env.device), len(ids), replacement=True)
    # Blend sources rather than imposing fixed class quotas on the global pool.
    if fraction > 0:
        pooled = torch.rand(len(ids), device=env.device) < fraction
        kinds[pooled] = 0
        pooled_recovery = pooled & (torch.rand(len(ids), device=env.device) < .8)
        positions = pooled_recovery.nonzero().flatten()
        if len(positions):
            states = sampler.draw(len(positions))
            env._recovery_state_id[ids[positions]] = states
            kinds[positions] = sampler.kinds[states]
    env._curriculum_kind[ids] = kinds
    env._curriculum_counts += torch.bincount(kinds, minlength=len(KINDS))
    for k, name in enumerate(KINDS[1:], 1):
        selected = ids[kinds == k]
        if not len(selected):
            continue
        if k <= 3:
            fraction = supine_probability if k == 1 else grounded_probability
            if fraction < 1:
                grounded = torch.rand(len(selected), device=env.device) < fraction
                airborne = selected[~grounded]
                if len(airborne):
                    reset_airborne_fall(env, airborne, k)
                    env._curriculum_airborne[airborne] = True
                    env._fall_source_counts[0] += len(airborne)
                selected = selected[grounded]
            env._fall_source_counts[1] += len(selected)
        if not len(selected):
            continue
        missing = selected[env._recovery_state_id[selected] < 0]
        if len(missing):
            env._recovery_state_id[missing] = sampler.draw(len(missing), name)
        index = env._recovery_state_id[selected]
        q = sampler.qpos[index].clone()
        q[:, :3] += env.scene.env_origins[selected]
        restore_recorded_state(env, q, sampler.qvel[index], selected)
    sampler.register_draws(ids)
    p = probabilities(iteration, getattr(env, '_supine_sampling_restore_start', None),
                      consolidation, sampler.kind_probabilities())
    log = env.extras.setdefault('log', {})
    log['Curriculum/iteration'] = iteration
    log['Curriculum/consolidation_progress'] = consolidation_progress(iteration, getattr(env, '_recovery_consolidation', None))
    log['Curriculum/grounded_fall_probability'] = grounded_probability
    log['Curriculum/supine_grounded_probability'] = supine_probability
    log['Curriculum/normal_reset_probability'] = p[0]
    log['Curriculum/supine_reset_probability'] = p[1]
    total = env._fall_source_counts.sum().clamp_min(1)
    log['Curriculum/airborne_fall_reset_share'] = env._fall_source_counts[0] / total
    log['Curriculum/grounded_fall_reset_share'] = env._fall_source_counts[1] / total
    for k, name in enumerate(KINDS):
        log[f'Curriculum/{name}_probability'] = p[k]
        log[f'Curriculum/{name}_reset_share'] = env._curriculum_counts[k] / env._curriculum_counts.sum().clamp_min(1)


def curriculum_frames(env, env_ids=None):
    if not hasattr(env, '_curriculum_kind'):
        return
    log = env.extras.setdefault('log', {})
    if hasattr(env, '_curriculum_airborne'):
        log['Curriculum/airborne_fall_frame_share'] = env._curriculum_airborne.float().mean()
        fallen_kind = (env._curriculum_kind >= 1) & (env._curriculum_kind <= 3)
        log['Curriculum/grounded_fall_frame_share'] = (fallen_kind & ~env._curriculum_airborne).float().mean()
        supine = env._curriculum_kind == 1
        log['Curriculum/supine_grounded_frame_share'] = (supine & ~env._curriculum_airborne).float().mean()
        log['Curriculum/supine_airborne_frame_share'] = (supine & env._curriculum_airborne).float().mean()
    for k, name in enumerate(KINDS):
        log[f'Curriculum/{name}_frame_share'] = (env._curriculum_kind == k).float().mean()
