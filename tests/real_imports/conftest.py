"""Configuration for real import tests - NO MOCKS"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# DO NOT import mock_dependencies here
# These tests should use real imports only