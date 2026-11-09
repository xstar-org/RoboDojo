"""Launch the standard client using its explicit environment-factory hook."""

import os
from pathlib import Path
import runpy
import sys

runtime = Path(__file__).resolve().parent
sys.path.insert(0, str(runtime))
# Native PhysX recovery re-execs main.py; keep its explicit hook importable.
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(runtime)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
)
entry = runtime / "RoboDojo/src/eval_client/main.py"
sys.argv[0] = str(entry)
sys.argv.extend(["--env_factory_module", "client_runtime"])
runpy.run_path(str(entry), run_name="__main__")
