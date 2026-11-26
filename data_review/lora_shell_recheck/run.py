"""Re-run the frozen case arguments and the additional table selection."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else Path(__file__).resolve().parent / "runs"
plans = [
    ("mvp_portability", Path("/tmp/mvp-inputs"), {"lora", "gpt2"}),
    ("independent_evaluation", Path("/tmp/ind-eval"), {"ddpm", "csd"}),
]
with tempfile.TemporaryDirectory(prefix="table-audit-") as temporary:
    work = Path(temporary)
    (work / "ai_auditor").symlink_to(root / "ai_auditor")
    jobs = []
    for manifest_directory, inputs, ids in plans:
        parent = root / "data_review" / manifest_directory
        manifest = parent / "cases.tsv"
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == (parent / "cases.sha256").read_text().split()[0]
        with manifest.open() as file:
            for row in csv.DictReader(file, delimiter="\t"):
                name = row["case_id"]
                if name not in ids:
                    continue
                pdf, repo = inputs / name / "paper.pdf", inputs / name / "git"
                assert hashlib.sha256(pdf.read_bytes()).hexdigest() == row["pdf_sha256"]
                actual = subprocess.check_output(
                    ["git", "-C", repo, "rev-parse", "--verify", row["commit"] + "^{commit}"], text=True)
                assert actual.strip() == row["commit"]
                target = work / "inputs" / name
                target.mkdir(parents=True)
                (target / "paper.pdf").symlink_to(pdf)
                (target / "repo.git").symlink_to(repo)
                argv = json.loads(row["cli_argv"])
                jobs.append((name, argv))
                if name == "lora":
                    shell = ["--table-shell-recipe", "examples/NLU/roberta_large_cola.sh"]
                    jobs.append(("lora-table", [*argv, "--paper-lr-table", "9", "RoBERTa-large LoRA", "CoLA"]))
                    jobs.append(("lora-shell", [*argv, "--paper-lr-table", "9", "RoBERTa-large LoRA", "CoLA", *shell]))
                    jobs.append(("lora-shell-dagger", [*argv, "--paper-lr-table", "9", "RoBERTa-large LoRA†", "CoLA", *shell]))
    out.mkdir()
    for name, argv in jobs:
        command = [sys.executable, "-E", "-s", "-m", "ai_auditor", *argv]
        result = subprocess.run(command, cwd=work, capture_output=True)
        (out / f"{name}.run.json").write_text(json.dumps({"argv": command, "exit_code": result.returncode}, indent=2) + "\n")
        (out / f"{name}.stderr.txt").write_bytes(result.stderr)
        (out / f"{name}.json").write_bytes(result.stdout)
        result.check_returncode()
        audit = json.loads(result.stdout)
        print(name, audit["status"], audit["paper_value"], audit["code_value"])
