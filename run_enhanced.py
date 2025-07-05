#!/usr/bin/env python
"""Run the enhanced Hunyuan3D Studio application"""

import subprocess
import sys

# Run the enhanced app
subprocess.run([sys.executable, "-m", "hunyuan3d_app.app_enhanced"] + sys.argv[1:])