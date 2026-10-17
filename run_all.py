"""Start the backend (server.py) and the voice agent (agent.py) with one command.

Usage:  python run_all.py            (agent in dev mode)
        python run_all.py --start    (agent in production "start" mode)
Press Ctrl+C to stop both. If one of them stops, the other is stopped too.
"""
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent
AGENT_MODE = "start" if "--start" in sys.argv else "dev"
COMMANDS = {
    "server": [sys.executable, "server.py"],
    "agent": [sys.executable, "agent.py", AGENT_MODE],
}


def main() -> None:
    procs = {name: subprocess.Popen(cmd, cwd=BASE_DIR) for name, cmd in COMMANDS.items()}
    try:
        while True:
            for name, proc in procs.items():
                code = proc.poll()
                if code is not None:
                    print(f"[run_all] {name} stopped (exit code {code}). Stopping everything.")
                    return
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[run_all] Stopping...")
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                proc.terminate()
        for proc in procs.values():
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    main()