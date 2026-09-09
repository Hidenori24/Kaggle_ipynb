import os
import sys

SUBMIT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "submit"))
if SUBMIT_DIR not in sys.path:
    sys.path.insert(0, SUBMIT_DIR)
