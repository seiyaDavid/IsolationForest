"""
Test Configuration Module

This module sets up the testing environment by:
    - Configuring Python path for test discovery
    - Setting up test fixtures
    - Providing common test utilities
"""

import os
import sys

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
