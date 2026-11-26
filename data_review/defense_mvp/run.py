"""Reproduce the frozen audit commands after retrieving the pinned inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

base = Path(__file__).resolve().parent


def sha(path):
    with path.open("rb") as fd:
        return hashlib.file_digest(fd, "sha256").hexdigest()


def verify(path, manifest):
    if sha(path) != manifest.read_text().split()[0]:
        raise SystemExit(f"SHA-256 mismatch: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Directory containing CASE/paper.pdf and CASE/git")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true", help="Retrieve missing PDF and Git objects first")
    args = parser.parse_args()
    archive = base / "cli_snapshot.tar.gz"
    manifest = base / "cases.tsv"
    verify(archive, base / "cli_snapshot.sha256")
    verify(manifest, base / "cases.sha256")
    python = subprocess.check_output([sys.executable, "--version"], text=True).strip()
    if python != (base / "runtime.txt").read_text().splitlines()[0]:
        raise SystemExit("Python version differs from the frozen environment.")
    lock = subprocess.check_output([sys.executable, "-m", "pip", "list", "--format=freeze"], text=True)
    if lock != (base / "runtime.lock.txt").read_text():
        raise SystemExit("Installed package versions differ from the frozen environment.")
    with manifest.open() as fd:
        rows = list(csv.DictReader(fd, delimiter="\t"))
    out = args.output.resolve()

    with tempfile.TemporaryDirectory(prefix="independent-auditor-") as tmp:
        work = Path(tmp)
        with tarfile.open(archive) as tar:
            tar.extractall(work, filter="data")
        for row in rows:
            name = row["case_id"]
            inputs = args.inputs.resolve() / name
            pdf, repo = inputs / "paper.pdf", inputs / "git"
            commit = row["commit"]
            if args.fetch:
                inputs.mkdir(parents=True, exist_ok=True)
                if not pdf.is_file():
                    subprocess.run(["curl", "-fLsS", row["pdf_url"], "-o", pdf], check=True)
                if not repo.is_dir():
                    subprocess.run(["git", "init", "--bare", repo], check=True)
                if subprocess.run(["git", "-C", repo, "cat-file", "-e", f"{commit}^{{commit}}"],
                                  capture_output=True).returncode:
                    subprocess.run(["git", "-C", repo, "fetch", "--depth=1", "--no-tags", row["repo_url"], commit],
                                   check=True)
            if sha(pdf) != row["pdf_sha256"]:
                raise SystemExit(f"PDF SHA-256 mismatch: {pdf}")
            actual = subprocess.check_output(
                ["git", "-C", repo, "rev-parse", "--verify", f"{commit}^{{commit}}"], text=True).strip()
            if actual != commit:
                raise SystemExit(f"Git commit mismatch: {repo}")
            expected_inputs = work / "inputs" / name
            expected_inputs.mkdir(parents=True)
            (expected_inputs / "paper.pdf").symlink_to(pdf)
            (expected_inputs / "repo.git").symlink_to(repo)
        out.mkdir(parents=True)
        for row in rows:
            name = row["case_id"]
            argv = [sys.executable, "-E", "-s", "-m", "ai_auditor", *json.loads(row["cli_argv"])]
            start = time.perf_counter()
            try:
                result = subprocess.run(argv, cwd=work, capture_output=True, timeout=60)
                stdout, stderr, code = result.stdout, result.stderr, result.returncode
            except subprocess.TimeoutExpired as error:
                stdout, stderr, code = error.stdout or b"", error.stderr or b"", None
            elapsed = time.perf_counter() - start
            try:
                parsed = json.loads(stdout)
                valid = isinstance(parsed, dict) and parsed.get("status") in {
                    "match", "potential_discrepancy", "insufficient_materials"
                }
            except (ValueError, UnicodeError):
                valid = False
            (out / f"{name}.{'json' if valid else 'stdout.txt'}").write_bytes(stdout)
            (out / f"{name}.stderr.txt").write_bytes(stderr)
            info = {"argv": argv, "exit_code": code, "launch_error": code != 0 or not valid, "seconds": elapsed,
                    "pdf_sha256": row["pdf_sha256"], "verified_commit": row["commit"]}
            (out / f"{name}.run.json").write_text(json.dumps(info, indent=2) + "\n")
    verify(manifest, base / "cases.sha256")


if __name__ == "__main__":
    main()
