"""Where Sheep keeps its own things (settings, model grades, herders) -- not the workspace."""
import os
from pathlib import Path

HOME = Path(os.environ.get("MLMI_HOME", Path.home() / ".mlmi"))
REPO = Path(__file__).resolve().parent.parent.parent       # the sheep repo
BUILTIN_HERDERS = REPO / "mlmi" / "herders"
