import sys
import os

## Useful for importing src modules in tests without needing to install the package

# Add project root to Python path so src module can be found
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))