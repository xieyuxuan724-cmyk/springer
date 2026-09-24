"""Export the repository's BENI assets and nominal mjlab actuator configuration."""
import ast
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SOURCE = REPO / 'assets/beni_base'
DEST = ROOT / '.generated/robot'
DEST.mkdir(parents=True, exist_ok=True)
# Read literal actuator parameters without importing the training/GPU stack.
tree = ast.parse((REPO / 'python/beni_rl/beni_robot.py').read_text())
motors = {}
for call in ast.walk(tree):
    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in ('BeniMotorCfg', 'IdealPdActuatorCfg'):
        values = {k.arg: ast.literal_eval(k.value) for k in call.keywords if k.arg}
        for name in values.pop('target_names_expr'):
            motors[name] = dict(values)
root = ET.parse(SOURCE / 'mjcf/beni.xml').getroot()
root.find('compiler').attrib.update(meshdir='meshes', texturedir='textures', autolimits='true')
# Match make_beni_velocity_flat_env_cfg / mjlab's velocity environment. The
# contact solver affects the wheel impulse and pitch during takeoff/landing.
ET.SubElement(root, 'option', timestep='0.002', gravity='0 0 -9.81', integrator='implicitfast',
              iterations='10', ls_iterations='20', ccd_iterations='50', cone='pyramidal')
old = root.find('actuator')
if old is not None: root.remove(old)
act = ET.SubElement(root, 'actuator')
for joint in root.findall('.//joint'):
    name = joint.get('name')
    if name not in motors: continue
    cfg = motors[name]
    joint.attrib.update(ref='0', actuatorfrclimited='false', armature=str(cfg['armature']), frictionloss=str(cfg['frictionloss']), damping='0')
    joint.attrib.pop('actuatorfrcrange', None)
    limit = cfg.get('simulation_effort_limit', cfg['effort_limit'])
    ET.SubElement(act, 'motor', name=name, joint=name, gear='1', ctrllimited='true', ctrlrange=f'{-limit} {limit}')
for geom in root.findall('.//geom'):
    if geom.get('contype', '1') != '0' or geom.get('conaffinity', '1') != '0':
        geom.set('group', '0')  # Reserve group 3 for track-only collision diagnostics.
for name in ('LW', 'RW'):
    body = root.find(f'.//body[@name="{name}_Link"]')
    for geom in body.findall('geom'):
        if geom.get('contype', '1') != '0' or geom.get('conaffinity', '1') != '0':
            geom.set('name', name + '_collision')
            geom.set('priority', '1')
files = []
for kind, directory in [('mesh', 'meshes'), ('texture', 'textures')]:
    for asset in root.findall(f'asset/{kind}'):
        name = asset.get('file')
        if not name: continue
        assert (SOURCE / directory / name).is_file(), f'Missing source asset: {directory}/{name}'
        files.append(f'{directory}/{name}')
# Each skin needs its own rgb atlas; take the list from game/skins.mjs so the
# export and the picker can never disagree, and fail loudly on a missing file.
skin_maps = re.findall(r"texture: '([^']+)'", (ROOT / 'game/skins.mjs').read_text(encoding='utf-8'))
assert len(skin_maps) == 8, f'expected 8 skin atlases, got {len(skin_maps)}'
for name in dict.fromkeys(skin_maps):
    if f'textures/{name}' in files: continue  # the mjcf already exported this one
    assert (SOURCE / 'textures' / name).is_file(), f'Missing skin: {name}'
    files.append(f'textures/{name}')
ET.indent(root)
ET.ElementTree(root).write(DEST / 'beni.xml', encoding='unicode')
(DEST / 'manifest.json').write_text(json.dumps(sorted(set(files)), indent=2) + '\n')
policy = REPO / 'assets/models/policy.onnx'
assert policy.is_file(), f'Missing source asset: {policy}'
# Remove generated asset copies left by older checkouts.
for directory in ('robot', 'policies'):
    shutil.rmtree(ROOT / 'public' / directory, ignore_errors=True)
(ROOT / 'public/favicon.svg').unlink(missing_ok=True)
# Browser calibration: 4 ms stays within mjlab's 4–10 ms command-delay range.
# Preserve nominal motor parameters; the browser defines its simplified spring.
config = {'name':'BENI', 'physicsDt':.002, 'decimation':5, 'commandDelaySteps':2, 'kneeDampingScale':1.2, 'jumpSupported':True, 'motors':motors, 'policySha256':hashlib.sha256(policy.read_bytes()).hexdigest()}
(ROOT / 'game/robot-config.json').write_text(json.dumps(config, indent=2) + '\n')
print('Prepared BENI metadata:', len(files), 'source assets; no public asset copies')
