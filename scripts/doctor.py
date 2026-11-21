# -*- coding: utf-8 -*-
"""Run the doctor self-check programmatically."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loopinfer.cli import cmd_doctor


class Args:
    pass


if __name__ == "__main__":
    sys.exit(cmd_doctor(Args()))
