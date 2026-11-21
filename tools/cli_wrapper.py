# -*- coding: utf-8 -*-
"""Tool: wrap the CLI as a callable for embedding in other apps."""

import sys

from loopinfer.cli import main


def run_cli(args):
    """Run the CLI with a list of string args, returns exit code."""
    return main(args)


if __name__ == "__main__":
    sys.exit(run_cli(sys.argv[1:]))
