# Copyright (c) 2026 Open Beni
# SPDX-License-Identifier: MIT

"""Physical recovery subgroups shared by collection, sampling and assessment."""

import hashlib
from pathlib import Path
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply

SUBGROUPS = ("base_front_free", "base_front_head", "base_rear_free", "base_rear_head",
             "legs_folded", "legs_asymmetric", "legs_extended")


def recovery_subgroups(env, sample):
    """End faces point down within 15 degrees; distinguish head-assisted support."""
    robot = env.scene['robot']
    sensor = env.scene['body_ground_contact']
    forces = sensor.data.force.norm(dim=-1) > .1
    def contact(suffix):
        return forces[:, [i for i,n in enumerate(sensor.primary_names) if n.endswith(suffix)]].any(-1)
    base = contact('base_link')
    head = contact('head')
    legs = contact(('1_Link', '2_Link'))
    wheels = sample['contacts'].any(-1)
    axis = torch.zeros_like(robot.data.root_link_pos_w)
    axis[:, 0] = 1
    vertical = quat_apply(robot.data.root_link_quat_w, axis)[:, 2]
    front = (vertical < -.9659258263) & base & ~legs & ~wheels
    rear = (vertical > .9659258263) & base & ~legs & ~wheels
    joint_ids = robot.find_joints(('L1', 'R1', 'L2', 'R2'), preserve_order=True)[0]
    q = robot.data.joint_pos[:, joint_ids]
    asymmetric = (q[:,0]-q[:,1]).abs() > .4
    asymmetric |= (q[:,2]-q[:,3]).abs() > .4
    folded = (q[:,2:] > 1.).all(-1) & ~asymmetric
    extended = (q[:,2:] < .8).all(-1) & ~asymmetric
    return torch.stack((front & ~head, front & head, rear & ~head, rear & head,
                        folded, asymmetric, extended), -1)


def load_recovery_bank(path):
    """Read arrays without pickles, rejecting malformed data before simulation."""
    from .recovery_curriculum import KINDS
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if 'subgroup_names' in data and tuple(data['subgroup_names']) != SUBGROUPS:
        raise ValueError('Recovery subgroup schema does not match this implementation')
    for name in KINDS[1:]:
        q, v = data[f'{name}_qpos'], data[f'{name}_qvel']
        if (q.ndim != 2 or v.ndim != 2 or not len(q) or len(q) != len(v)
                or q.shape[1] != 15 or v.shape[1] != 14
                or not np.isfinite(q).all() or not np.isfinite(v).all()
                or not np.allclose(np.linalg.norm(q[:,3:7], axis=1), 1., atol=1e-4)):
            raise ValueError(f'Invalid recovery states in {path}: {name}')
        if f'{name}_tags' in data and (data[f'{name}_tags'].shape != (len(q), len(SUBGROUPS))
                                      or data[f'{name}_tags'].dtype != np.bool_):
            raise ValueError(f'Invalid recovery subgroup labels: {name}')
    return data


def bank_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def update_hard_states(failures, groups, result):
    """Retain states failing/slower than 3s in two training-side checks.

    Deduplicate overlapping broad/subgroup trials within a single check. Membership
    only grows during a run; held-out results must never be passed to this function.
    """
    candidates = {}
    for scenario in result['scenarios'].values():
        for row in scenario['trials']:
            if row['valid'] and (not row['recovered'] or row['recovery_s'] > 3.):
                candidates[(row['kind'],row['index'])] = row
    for (kind,index),row in candidates.items():
        key = f'{kind}/{index}'
        failures[key] = failures.get(key,0)+1
        if failures[key] < 2: continue
        by_kind = groups.setdefault(kind,{})
        for subgroup in row['subgroups'] or ('other',):
            by_kind[subgroup] = sorted(set(by_kind.get(subgroup,())) | {index})


def sample_bank_indices(size, count, device, hard_groups=()):
    """Half uniform coverage, half uniform hard subgroup then uniform state."""
    indices = torch.randint(size,(count,),device=device)
    if not hard_groups: return indices
    selected = (torch.rand(count,device=device) < .5).nonzero().flatten()
    group_ids = torch.randint(len(hard_groups),(len(selected),),device=device)
    for i,group in enumerate(hard_groups):
        positions = selected[group_ids == i]
        indices[positions] = group[torch.randint(len(group),(len(positions),),device=device)]
    return indices
