"""Render the shipped BENI in MuJoCo as the reference for the web renderer.

The web viewer has to agree with what MuJoCo shows, so this renders the same
`assets/beni_base/mjcf/beni.xml` with one of the eight colour atlases and writes
PNGs next to the browser screenshots for side-by-side comparison.

    python scripts/mujoco-reference.py            # every skin, two views
    python scripts/mujoco-reference.py red white  # only these skins

MuJoCo reads PNG textures only, so a JPG atlas is converted into a temp dir
first, and only the rgb layer is swapped: normal/orm/emissive never reach
MuJoCo's renderer, which is why the web defaults to the colour map alone.
"""

import pathlib
import re
import shutil
import sys
import tempfile

import mujoco
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = ROOT.parent
ASSETS = REPO / 'assets/beni_base'
OUT = ROOT / 'outputs'
# Matched to the debug view's framing: three quarters from the front and rear.
VIEWS = {'front': (120, -15, 0.95), 'rear': (-40, -20, 0.95)}
# Two lights, because beni.xml carries none of its own.
LIGHTS = (
    '<light pos="1.2 -1.6 2.2" dir="-0.4 0.55 -0.75" diffuse="1.1 1.1 1.1"'
    ' specular="0.4 0.4 0.4"/>'
    '<light pos="-1.4 1.2 1.6" dir="0.5 -0.45 -0.7" diffuse="0.5 0.55 0.6"/>'
)


def skins():
    """The picker's skin -> atlas mapping, read from the one source of truth."""
    text = (ROOT / 'game/skins.mjs').read_text(encoding='utf-8')
    ids = re.findall(r"id: '([^']+)'", text)
    files = re.findall(r"texture: '([^']+)'", text)
    return dict(zip(ids, files))


def render(skin, atlas, tmp):
    texture = tmp / f'{skin}.png'
    Image.open(ASSETS / 'textures' / atlas).convert('RGB').save(texture)
    xml = (ASSETS / 'mjcf/beni.xml').read_text(encoding='utf-8')
    meshes, textures = (ASSETS / 'meshes').as_posix(), (ASSETS / 'textures').as_posix()
    xml = xml.replace('meshdir="../meshes"', f'meshdir="{meshes}"')
    xml = xml.replace('texturedir="../textures"', f'texturedir="{textures}"')
    xml = xml.replace('file="beni_white.png"', f'file="{texture.as_posix()}"')
    # The offscreen framebuffer defaults to 640 px, narrower than the render.
    xml = xml.replace(
        '<compiler',
        '<visual><global offwidth="700" offheight="700"/></visual><compiler',
        1,
    )
    xml = xml.replace('<worldbody>', '<worldbody>' + LIGHTS, 1)
    path = tmp / f'beni_{skin}.xml'
    path.write_text(xml, encoding='utf-8')
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for name, (azimuth, elevation, distance) in VIEWS.items():
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
        camera.lookat[:] = [0, 0, 0.22]
        with mujoco.Renderer(model, 700, 700) as renderer:
            renderer.update_scene(data, camera)
            out = OUT / f'mujoco-{skin}-{name}.png'
            Image.fromarray(renderer.render()).save(out)
            print('wrote', out.relative_to(ROOT))


def main():
    OUT.mkdir(exist_ok=True)
    available = skins()
    wanted = sys.argv[1:] or list(available)
    unknown = [s for s in wanted if s not in available]
    assert not unknown, f'unknown skins {unknown}; have {list(available)}'
    tmp = pathlib.Path(tempfile.mkdtemp(prefix='beni-reference-'))
    try:
        for skin in wanted:
            render(skin, available[skin], tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


main()
