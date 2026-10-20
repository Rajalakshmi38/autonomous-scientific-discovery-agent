"""Probes that check the sandbox restrictions on a REAL Docker engine.

Run with:  python -m src.main sandbox-test
Each probe is a tiny script that TRIES to do something forbidden. A probe passes only when the
sandbox stopped it. Until all probes pass on your machine, do not describe the sandbox as secure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from src.execution.sandbox import DockerSandbox, ExecutionResult, Status


@dataclass
class Probe:
    name: str
    what: str
    code: str
    check: Callable[[ExecutionResult], bool]
    timeout: float = 30.0


@dataclass
class ProbeResult:
    name: str
    what: str
    passed: bool
    detail: str


def _out(r: ExecutionResult) -> str:
    return r.stdout.strip()


PROBES: List[Probe] = [
    Probe("python_runs", "a normal script runs and prints", "print(6 * 7)",
          lambda r: r.ok and _out(r) == "42"),
    Probe("non_root_user", "the script does not run as root",
          "import os\nprint(os.getuid())",
          lambda r: r.ok and _out(r) not in ("", "0")),
    Probe("no_network", "outgoing network connections and DNS fail", """\
import socket
blocked = 0
try:
    socket.create_connection(("1.1.1.1", 53), timeout=3)
except OSError:
    blocked += 1
try:
    socket.gethostbyname("example.com")
except OSError:
    blocked += 1
print("BLOCKED" if blocked == 2 else "REACHED")
""", lambda r: r.ok and _out(r) == "BLOCKED"),
    Probe("read_only_filesystem", "the container's system files cannot be modified", """\
try:
    with open("/etc/sandbox_probe", "w") as fh:
        fh.write("x")
    print("WRITABLE")
except OSError:
    print("READ_ONLY")
""", lambda r: r.ok and _out(r) == "READ_ONLY"),
    Probe("tmp_scratch_space", "a small writable /tmp exists for scratch files", """\
with open("/tmp/probe.txt", "w") as fh:
    fh.write("ok")
print("TMP_OK")
""", lambda r: r.ok and _out(r) == "TMP_OK"),
    Probe("no_host_environment", "host environment variables are not passed in", """\
import os
leaked = [k for k in os.environ if k.upper() in ("USERPROFILE", "APPDATA", "COMPUTERNAME", "USERNAME", "LLM_PROVIDER", "OLLAMA_MODEL")]
print("LEAK" if leaked else "CLEAN")
""", lambda r: r.ok and _out(r) == "CLEAN"),
    Probe("no_host_files", "no host drives or folders are visible", """\
import os
mnt = os.listdir("/mnt") if os.path.isdir("/mnt") else []
print("CLEAN" if not mnt and not os.path.exists("/host_mnt") else "VISIBLE:" + ",".join(mnt))
""", lambda r: r.ok and _out(r) == "CLEAN"),
    Probe("memory_limit", "a script that eats memory is stopped", """\
chunks = []
while True:
    chunks.append(bytearray(b"x") * (50 * 1024 * 1024))
""", lambda r: r.status == Status.OUT_OF_MEMORY or r.exit_code == 137, timeout=40),
    Probe("process_limit", "a fork bomb is limited", """\
import os, time
children = []
try:
    for _ in range(400):
        pid = os.fork()
        if pid == 0:
            time.sleep(5)
            os._exit(0)
        children.append(pid)
    print("UNLIMITED")
except OSError:
    print("LIMITED")
""", lambda r: r.ok and _out(r) == "LIMITED"),
    Probe("timeout_enforced", "an endless loop is stopped by the time limit",
          "while True:\n    pass\n",
          lambda r: r.status == Status.TIMEOUT, timeout=5),
]


def run_self_test(sandbox: DockerSandbox) -> List[ProbeResult]:
    """Run every probe, then check that no containers were left behind."""
    results: List[ProbeResult] = []
    for probe in PROBES:
        r = sandbox.run_code(probe.code, timeout_seconds=probe.timeout)
        passed = bool(probe.check(r))
        detail = f"status={r.status}, exit={r.exit_code}, output={_out(r)[:60]!r}"
        if r.status == Status.SANDBOX_UNAVAILABLE:
            detail = r.message
        results.append(ProbeResult(probe.name, probe.what, passed, detail))
    leftovers = sandbox.leftover_containers()
    results.append(ProbeResult(
        "containers_cleaned_up", "no sandbox containers are left running or stopped",
        not leftovers, "none left" if not leftovers else f"left behind: {leftovers}"))
    return results
