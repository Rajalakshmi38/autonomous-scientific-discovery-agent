"""Docker sandbox for running UNTRUSTED, model-written Python code.

Design (every point is checked by ``python -m src.main sandbox-test`` on a real Docker engine):
  * The script is sent through stdin. No host folder is ever mounted, so the code cannot see your files.
  * No network (``--network none``), read-only root filesystem, small writable /tmp, no Linux
    capabilities, no privilege escalation, non-root user, memory / CPU / process limits.
  * A wall-clock timeout. Killing only the ``docker`` client would leave the code running inside the
    container, so on timeout we kill the CONTAINER itself, then always remove it.
  * Captured output is size-capped so a script that prints forever cannot exhaust host memory.

IMPORTANT: Do not call this sandbox "secure" until ``sandbox-test`` has passed on your machine.
Docker containers are strong isolation for this use, but they are not a perfect security boundary.
"""
from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CONTAINER_PREFIX = "asda-sbx-"


class Status:
    """Execution outcomes. Only the first four are results of the code itself."""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    SANDBOX_UNAVAILABLE = "SANDBOX_UNAVAILABLE"   # Docker missing/stopped, or image not built
    SANDBOX_ERROR = "SANDBOX_ERROR"               # Docker itself failed in an unexpected way
    NOT_RUN = "NOT_RUN"

    REPAIRABLE = {FAILED, TIMEOUT, OUT_OF_MEMORY}   # a code fix could help; retrying SANDBOX_* cannot


@dataclass(frozen=True)
class SandboxConfig:
    image: str = "asda-sandbox:latest"
    timeout_seconds: float = 30.0
    memory_mb: int = 512
    cpus: float = 1.0
    pids_limit: int = 64
    max_output_bytes: int = 64 * 1024
    max_code_bytes: int = 100_000
    docker_command: Tuple[str, ...] = ("docker",)

    @classmethod
    def from_settings(cls, settings) -> "SandboxConfig":
        return cls(
            image=settings.sandbox_image,
            timeout_seconds=settings.sandbox_timeout_seconds,
            memory_mb=settings.sandbox_memory_mb,
            cpus=settings.sandbox_cpus,
            pids_limit=settings.sandbox_pids_limit,
            max_output_bytes=settings.sandbox_max_output_kb * 1024,
            docker_command=tuple(shlex.split(settings.docker_command, posix=(os.name != "nt"))),
        )


@dataclass
class ExecutionResult:
    status: str
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.0
    error_kind: Optional[str] = None      # missing_dependency | syntax_error | runtime_error | timeout | ...
    output_truncated: bool = False
    message: str = ""                     # plain-English explanation for the user
    container_name: str = ""

    @property
    def ok(self) -> bool:
        return self.status == Status.SUCCESS

    @property
    def repairable(self) -> bool:
        return self.status in Status.REPAIRABLE

    def to_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


@dataclass
class SandboxStatus:
    ok: bool
    docker_found: bool = False
    engine_running: bool = False
    image_present: bool = False
    message: str = ""


# --------------------------------------------------------------------------- helpers
def build_run_command(config: SandboxConfig, name: str) -> List[str]:
    """The exact ``docker run`` command. Note: no -v/--mount/--privileged anywhere (tested)."""
    mem = f"{config.memory_mb}m"
    return [
        *config.docker_command, "run",
        "--name", name,
        "-i",                                   # read the script from stdin
        "--pull", "never",                      # never fetch an image with this name from a registry
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--memory", mem, "--memory-swap", mem,  # no swap, so the memory limit is a hard limit
        "--cpus", str(config.cpus),
        "--pids-limit", str(config.pids_limit),
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--user", "10001:10001",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        config.image, "python", "-",
    ]


_MISSING_MODULE_RE = re.compile(r"No module named ['\"]([\w\.]+)['\"]")
_DOCKER_DOWN_HINTS = (
    "cannot connect to the docker daemon", "error during connect", "failed to connect to the docker api",
    "is the docker daemon running", "dockerdesktoplinuxengine", "docker daemon is not running",
)
_IMAGE_MISSING_HINTS = ("unable to find image", "no such image", "pull access denied", "not found: manifest")


def missing_module(stderr: str) -> Optional[str]:
    match = _MISSING_MODULE_RE.search(stderr or "")
    return match.group(1) if match else None


def classify_error(stderr: str) -> str:
    text = stderr or ""
    if "ModuleNotFoundError" in text or "No module named" in text:
        return "missing_dependency"
    if "SyntaxError" in text or "IndentationError" in text:
        return "syntax_error"
    if "MemoryError" in text:
        return "memory_error"
    if "Traceback (most recent call last)" in text:
        return "runtime_error"
    return "unknown_error"


class _Capture:
    """Reads a pipe on a background thread, keeping at most ``limit`` bytes."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.data = bytearray()
        self.truncated = False
        self._thread: Optional[threading.Thread] = None

    def start(self, stream) -> None:
        def work() -> None:
            try:
                while True:
                    chunk = stream.read(4096)
                    if not chunk:
                        return
                    room = self.limit - len(self.data)
                    if room > 0:
                        self.data.extend(chunk[:room])
                    if len(chunk) > room:
                        self.truncated = True       # keep draining so the process never blocks
            except (OSError, ValueError):
                return
        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()

    def finish(self, wait: float = 5.0) -> str:
        if self._thread is not None:
            self._thread.join(wait)
        return bytes(self.data).decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- sandbox
class DockerSandbox:
    def __init__(self, config: Optional[SandboxConfig] = None) -> None:
        self.config = config or SandboxConfig()

    # ---- small docker helpers -------------------------------------------------
    def _docker(self, args: List[str], timeout: float = 15.0) -> Optional[subprocess.CompletedProcess]:
        """Run a short docker command; None if docker is missing or hung."""
        try:
            return subprocess.run(
                [*self.config.docker_command, *args],
                capture_output=True, text=True, timeout=timeout, errors="replace",
            )
        except (OSError, subprocess.SubprocessError):
            return None

    def check_available(self) -> SandboxStatus:
        info = self._docker(["info", "--format", "{{.ServerVersion}}"], timeout=20)
        if info is None:
            return SandboxStatus(ok=False, message="Docker command not found. Install Docker Desktop and reopen the terminal.")
        if info.returncode != 0:
            return SandboxStatus(ok=False, docker_found=True, engine_running=False,
                                 message="Docker is installed but its engine is not running. "
                                         "Start Docker Desktop and wait until it says the engine is running.")
        image = self._docker(["image", "inspect", self.config.image, "--format", "{{.Id}}"])
        if image is None or image.returncode != 0:
            return SandboxStatus(ok=False, docker_found=True, engine_running=True, image_present=False, message=(
                f"Docker is running but the image '{self.config.image}' is not built yet. "
                "Build it once with: python -m src.main sandbox-build"))
        return SandboxStatus(ok=True, docker_found=True, engine_running=True, image_present=True,
                             message=f"Docker is running and image '{self.config.image}' is ready.")

    def build_image(self, context_dir: Path) -> int:
        """Build the sandbox image. Output goes straight to the console so you can see progress."""
        try:
            return subprocess.run(
                [*self.config.docker_command, "build", "-t", self.config.image, str(context_dir)]
            ).returncode
        except OSError:
            return 127

    def leftover_containers(self) -> List[str]:
        res = self._docker(["ps", "-a", "--filter", f"name={CONTAINER_PREFIX}", "--format", "{{.Names}}"])
        if res is None or res.returncode != 0:
            return []
        return [line.strip() for line in res.stdout.splitlines() if line.strip()]

    def _was_oom_killed(self, name: str) -> bool:
        res = self._docker(["inspect", "-f", "{{.State.OOMKilled}}", name], timeout=10)
        return bool(res and res.returncode == 0 and res.stdout.strip().lower() == "true")

    def _kill_container(self, name: str) -> None:
        self._docker(["kill", name], timeout=15)

    def _remove_container(self, name: str) -> None:
        self._docker(["rm", "-f", name], timeout=20)

    # ---- main entry ---------------------------------------------------------------
    def run_code(self, code: str, timeout_seconds: Optional[float] = None) -> ExecutionResult:
        timeout = timeout_seconds if timeout_seconds is not None else self.config.timeout_seconds
        if not isinstance(code, str) or not code.strip():
            return ExecutionResult(Status.FAILED, error_kind="invalid_input", message="There is no code to run.")
        encoded = code.encode("utf-8")
        if len(encoded) > self.config.max_code_bytes:
            return ExecutionResult(Status.FAILED, error_kind="invalid_input",
                                   message=f"The script is larger than {self.config.max_code_bytes} bytes.")

        name = f"{CONTAINER_PREFIX}{uuid.uuid4().hex[:10]}"
        command = build_run_command(self.config, name)
        started = time.monotonic()
        try:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            return ExecutionResult(Status.SANDBOX_UNAVAILABLE, error_kind="docker_missing", container_name=name,
                                   message=f"Could not start the Docker command ({exc}). Is Docker installed?")

        out, err = _Capture(self.config.max_output_bytes), _Capture(self.config.max_output_bytes)
        out.start(proc.stdout)
        err.start(proc.stderr)
        try:
            proc.stdin.write(encoded)
            proc.stdin.close()
        except OSError:
            pass    # docker exited before reading the script; its stderr explains why

        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            logger.warning("Sandbox timeout after %.0fs; killing container %s", timeout, name)
            self._kill_container(name)      # stops the code itself, not just the docker client
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.error("docker client did not exit after kill")
        stdout, stderr = out.finish(), err.finish()
        duration = time.monotonic() - started

        try:
            oom = self._was_oom_killed(name)
        finally:
            self._remove_container(name)    # always clean up, whatever happened

        result = self._interpret(proc.returncode, timed_out, oom, stdout, stderr, timeout)
        result.duration_seconds = round(duration, 2)
        result.output_truncated = out.truncated or err.truncated
        result.container_name = name
        logger.info("Sandbox run %s: %s (exit=%s, %.1fs)", name, result.status, result.exit_code, duration)
        return result

    def _interpret(self, code: Optional[int], timed_out: bool, oom: bool,
                   stdout: str, stderr: str, timeout: float) -> ExecutionResult:
        base = dict(exit_code=code, stdout=stdout, stderr=stderr)
        if timed_out:
            return ExecutionResult(Status.TIMEOUT, error_kind="timeout", **base,
                                   message=f"The script ran longer than {timeout:.0f} seconds and was stopped.")
        if code == 0:
            return ExecutionResult(Status.SUCCESS, **base, message="The script finished normally.")
        if oom:
            return ExecutionResult(Status.OUT_OF_MEMORY, error_kind="out_of_memory", **base,
                                   message=f"The script used more than {self.config.memory_mb} MB of memory and was stopped.")
        lowered = stderr.lower()
        from_docker = "traceback (most recent call last)" not in lowered   # python errors are the script's own
        if from_docker and any(h in lowered for h in _DOCKER_DOWN_HINTS):
            return ExecutionResult(Status.SANDBOX_UNAVAILABLE, error_kind="docker_not_running", **base,
                                   message="Docker's engine is not running. Start Docker Desktop and try again.")
        if from_docker and any(h in lowered for h in _IMAGE_MISSING_HINTS):
            return ExecutionResult(Status.SANDBOX_UNAVAILABLE, error_kind="image_missing", **base,
                                   message=f"The sandbox image '{self.config.image}' is missing. "
                                           "Build it with: python -m src.main sandbox-build")
        if from_docker and code == 125:
            return ExecutionResult(Status.SANDBOX_ERROR, error_kind="docker_error", **base,
                                   message="Docker could not start the container (see stderr).")
        kind = classify_error(stderr)
        module = missing_module(stderr)
        message = {
            "missing_dependency": f"The script needs a library that is not installed in the sandbox"
                                  f"{f' ({module})' if module else ''}. Only the standard library, numpy, scipy, pandas and scikit-learn are available.",
            "syntax_error": "The script has a Python syntax error.",
            "runtime_error": "The script raised an error while running.",
        }.get(kind, f"The script exited with code {code}.")
        return ExecutionResult(Status.FAILED, error_kind=kind, **base, message=message)
