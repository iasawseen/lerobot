#!/bin/bash
# Orin LIBERO eval entrypoint (companion to docker/Dockerfile.orin.libero). Writes
# ~/.libero/config.yaml and points get_assets_path() at the staged asset bundle
# (mounted at /libero-assets) BEFORE any `import libero` — libero/__init__.py calls
# input() if config.yaml is missing, which hangs headless. Then runs lerobot-eval.
# All eval CLI args are forwarded. Fully offline (run with HF_HUB_OFFLINE=1).
set -e
python3 - <<'PY'
import os, importlib.util
spec = importlib.util.find_spec("libero")
LIBERO_DIR = os.path.join(os.path.dirname(spec.origin), "libero")
# get_assets_path() returns <pkg>/libero/assets if it exists, else tries to download.
# Symlink the staged bundle there so it resolves with zero network.
link = os.path.join(LIBERO_DIR, "assets")
if not os.path.exists(link):
    try:
        os.symlink("/libero-assets", link)
    except FileExistsError:
        pass
d = os.path.expanduser("~/.libero")
os.makedirs(d, exist_ok=True)
lines = [
    "assets: /libero-assets",
    f"bddl_files: {os.path.join(LIBERO_DIR, 'bddl_files')}",
    f"init_states: {os.path.join(LIBERO_DIR, 'init_files')}",
    f"datasets: {os.path.join(LIBERO_DIR, '..', 'datasets')}",
]
with open(os.path.join(d, "config.yaml"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("LIBERO_DIR:", LIBERO_DIR)
print("assets symlink:", link, "->", os.path.realpath(link))
print("config.yaml:", lines)
PY
exec python3 -m lerobot.scripts.lerobot_eval "$@"
