import os
import sys
import argparse
try:
    import deepqc
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")


def cli(args=None):
    parser = argparse.ArgumentParser(
                description="deepqc")
    parser.add_argument("command", 
                        help="specify the sub-command to run, possible choices: "
                             "train, test, scf, stat, iterate")
    parser.add_argument("args", nargs=argparse.REMAINDER,
                        help="arguments to be passed to the sub-command")

    args = parser.parse_args(args)
    if args.command.upper() == "TRAIN":
        from deepqc.train.main import cli as subcli
    elif args.command.upper() == "TEST":
        from deepqc.train.test import cli as subcli
    elif args.command.upper() == "SCF":
        from deepqc.scf.main import cli as subcli
    elif args.command.upper() == "STAT":
        from deepqc.scf.tools import cli as subcli
    elif args.command.upper().startswith("ITER"):
        from deepqc.iterate.main import cli as subcli
    else:
        return ValueError(f"unsupported sub-command: {args.command}")
    
    subcli(args.args)


if __name__ == "__main__":
    cli()