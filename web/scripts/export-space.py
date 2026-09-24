"""Export complete public source and built CDN assets, never repository history."""
import argparse
import re
import shutil
import subprocess
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
ROOT_FILES = {"README.md", "LICENSE", "Dockerfile", ".dockerignore", ".gitignore", ".gitattributes", ".python-version", "pyproject.toml", "uv.lock"}
SOURCE_DIRS = {"assets", "python", "web"}
BLOCKED = {".git", ".codex", ".agents", ".openai", "node_modules", ".venv", "__pycache__", "tmp", "logs", "outputs", "dist", ".generated", "AGENTS.md", "MEMORY.md"}


def export(destination, space):
    if destination.exists():
        raise SystemExit("Use a new export directory; existing directories are not overwritten.")
    if not (ROOT / "web/dist/index.html").is_file():
        raise SystemExit("Build web/ before exporting.")
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    # HF manages LFS attributes separately from the source repository. Overwriting
    # them makes existing binary assets appear as pointer text during Docker builds.
    attributes_url = f"https://huggingface.co/spaces/{quote(space, safe='/')}/raw/main/.gitattributes"
    with urlopen(attributes_url, timeout=30) as response:
        space_attributes = response.read()
    selected = []
    for name in sorted(set(filter(None, paths))):
        path = Path(name)
        if name == ".gitattributes":
            continue
        if len(path.parts) == 1 and name not in ROOT_FILES:
            continue
        if len(path.parts) > 1 and path.parts[0] not in SOURCE_DIRS:
            continue
        if any(part in BLOCKED or part.startswith('.env') for part in path.parts):
            raise SystemExit(f"Unexpected private/generated file in source list: {name}")
        source = ROOT / path
        if not source.exists():
            continue
        if source.is_symlink():
            raise SystemExit(f"Unexpected source symlink: {name}")
        if source.suffix.lower() in {'.pem', '.key', '.log', '.mp4', '.webm'}:
            raise SystemExit(f"Unexpected private/recording file: {name}")
        selected.append(path)
    for path in selected:
        output = destination / path
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, output)
    (destination / '.gitattributes').write_bytes(space_attributes)
    shutil.copytree(ROOT / 'web/dist', destination / 'dist')
    # Build output can contain bundled dependencies, but never local source maps or history.
    for path in destination.rglob('*'):
        if not path.is_file():
            continue
        if path.suffix == '.map':
            raise SystemExit(f"Unexpected source map: {path.relative_to(destination)}")
        if path.suffix in {'.ts', '.tsx', '.js', '.mjs', '.json', '.md', '.py', '.toml', '.html'}:
            text = path.read_text(errors='replace')
            if re.search(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|hf_[A-Za-z0-9]{25,}|glpat-[A-Za-z0-9_-]{15,}|/home/[^/\s]+/code/', text):
                raise SystemExit(f"Private content requires review: {path.relative_to(destination)}")
    print(f"Exported {len(selected)} source files plus built CDN assets. No Git history included.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--space', default='mondo-robotics/open-beni', help='Existing public Space whose LFS attributes must be preserved')
    args = parser.parse_args()
    export(args.destination.resolve(), args.space)
