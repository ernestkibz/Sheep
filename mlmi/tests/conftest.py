"""Every test run gets its own Sheep home. paths.HOME is read once, at the first
import of the backend, so this must run before any test module imports it --
otherwise herders, chats and config made by tests land in the person's ~/.mlmi."""
import os
import tempfile

os.environ["MLMI_HOME"] = tempfile.mkdtemp(prefix="mlmi-test-home-")
