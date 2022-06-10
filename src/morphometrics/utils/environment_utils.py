import os
import platform

# set to true if running on github actions
on_ci = os.getenv("CI") == "true"

# set to true if running on windows
on_windows = platform.system() == "Windows"
