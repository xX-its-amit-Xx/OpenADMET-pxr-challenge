"""Windows-safe ipykernel launcher.

Fixes two issues on Windows:
1. asyncio ProactorEventLoop doesn't support add_reader (needed by zmq).
2. A stale VIRTUAL_ENV env var can confuse sub-processes.
"""
import asyncio
import os
import sys

# Fix ZMQ/zmq._future warning on Windows before any network code runs.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Clear the stale VIRTUAL_ENV that points to the system Python, not our venv.
os.environ.pop("VIRTUAL_ENV", None)

from ipykernel import kernelapp

kernelapp.launch_new_instance()
