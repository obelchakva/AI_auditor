"""Command line entry point for the initial learning-rate check."""

import argparse
import json
from pathlib import Path

from .initial_lr import audit


def main():
    parser = argparse.ArgumentParser(description="Compare the initial learning rate in a PDF and a Git commit.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    if not args.stage.strip():
        parser.error("--stage must contain a stage description")
    try:
        result = audit(args.pdf, args.repo, args.commit, args.stage.strip())
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
