import os
import sys
try:
    import deepks
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from deepks.main import scf_cli

if __name__ == "__main__":
    scf_cli()