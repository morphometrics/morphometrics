import os

# set to true if running on github actions
on_ci = os.getenv("CI") == "true"
