"""A FAKE `docker` command used only by the tests. It is NOT Docker and provides NO isolation.

It lets the tests exercise our real subprocess handling (stdin, threads, timeouts, output caps,
cleanup calls) on any machine. Whether the real sandbox restrictions work can only be shown by
running against real Docker: `python -m src.main sandbox-test`.

Behaviour is selected by markers in the submitted script, or by environment variables.
Every call is appended to the file named in FAKE_DOCKER_LOG.
"""
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

STATE_DIR = Path(tempfile.gettempdir()) / "asda_fake_docker"
STATE_DIR.mkdir(exist_ok=True)


def log(args):
    path = os.environ.get("FAKE_DOCKER_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(" ".join(args) + "\n")


def container_name(args):
    return args[args.index("--name") + 1] if "--name" in args else "unnamed"


def main():
    args = sys.argv[1:]
    log(args)
    if not args:
        return 2
    cmd = args[0]

    if cmd == "info":
        if os.environ.get("FAKE_DOCKER_DOWN") == "1":
            sys.stderr.write("failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine\n")
            return 1
        print("29.0.0")
        return 0
    if cmd == "image":
        return 1 if os.environ.get("FAKE_NO_IMAGE") == "1" else 0
    if cmd == "kill":
        return 0
    if cmd == "rm":
        (STATE_DIR / f"{args[-1]}.json").unlink(missing_ok=True)
        return 0
    if cmd == "inspect":
        state = STATE_DIR / f"{args[-1]}.json"
        oom = json.loads(state.read_text()).get("oom", False) if state.exists() else False
        print("true" if oom else "false")
        return 0
    if cmd == "ps":
        for f in STATE_DIR.glob("asda-sbx-*.json"):
            print(f.stem)
        return 0
    if cmd != "run":
        return 2

    name = container_name(args)
    code = sys.stdin.buffer.read().decode("utf-8")
    state = STATE_DIR / f"{name}.json"
    state.write_text(json.dumps({"oom": "#FAKE_OOM" in code}))
    if "#FAKE_DOCKER_DOWN" in code:
        sys.stderr.write("failed to connect to the docker API at npipe:////./pipe/dockerDesktopLinuxEngine\n")
        return 1
    if "#FAKE_NO_IMAGE" in code:
        sys.stderr.write("Unable to find image 'asda-sandbox:latest' locally\n")
        return 125
    if "#FAKE_OOM" in code:
        return 137
    try:
        exec(compile(code, "<sandbox>", "exec"), {"__name__": "__main__"})
    except SystemExit as exc:
        sys.stdout.flush()
        return exc.code if isinstance(exc.code, int) else 0
    except BaseException:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
