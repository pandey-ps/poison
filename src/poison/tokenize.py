import json
import os
import shutil
import subprocess


def _bindir() -> str:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, "bin")


def tokenize(model: str, text: str) -> list[int]:
    bd = _bindir()
    if not shutil.which("llama-tokenize") and not os.path.exists(os.path.join(bd, "llama-tokenize")):
        raise RuntimeError("llama-tokenize not found (add bin/ to PATH)")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = bd + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["PATH"] = bd + os.pathsep + env.get("PATH", "")
    r = subprocess.run(["llama-tokenize", "-m", model, "--ids", "--no-bos", "-p", text],
                       check=True, capture_output=True, text=True, env=env)
    for line in reversed(r.stdout.splitlines()):
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            return [int(x) for x in json.loads(line)]
    raise RuntimeError(f"cannot parse tokens for {text!r}: {r.stdout[-500:]}")
