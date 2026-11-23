import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pymupdf


class AuditTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root / "research"
        self.repo.mkdir()
        self.pdf = self.root / "article.pdf"
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Test")
        self.write("README.md", "Step 1: run `python fit.py`.\nStep 2: run `python refine.py`.\n")
        self.write("refine.py", "optimizer = SGD(model.parameters(), lr=0.2)\n")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True).strip()

    def write(self, path, source):
        (self.repo / path).write_text(source, encoding="utf-8")

    def commit(self):
        self.git("add", ".")
        self.git("-c", "core.hooksPath=/dev/null", "commit", "-qm", "Test snapshot")
        return self.git("rev-parse", "HEAD")

    def article(self, text):
        with pymupdf.open() as doc:
            doc.new_page().insert_text((72, 72), "Methods")
            page = doc.new_page()
            page.insert_textbox(pymupdf.Rect(60, 72, 520, 400), text)
            doc.save(self.pdf)

    def run_audit(self, commit, stage="Step 1"):
        process = subprocess.run([
            sys.executable, "-m", "ai_auditor", "--pdf", str(self.pdf), "--repo", str(self.repo),
            "--commit", commit, "--stage", stage,
        ], check=True, capture_output=True, text=True)
        return json.loads(process.stdout)

    def test_discrepancy_uses_imported_class_constant_and_parameter_groups(self):
        self.article("In Step 1, the learning rate was initially set to 0.1.")
        self.write("settings.py", "class Settings:\n    RATE = 0.05\n\nsettings = Settings()\n")
        self.write("fit.py", """from settings import settings
def train():
    optimizer = optim.SGD([
        {'params': first.parameters(), 'lr': settings.RATE},
        {'params': second.parameters(), 'lr': settings.RATE},
    ])
if __name__ == '__main__':
    train()
""")
        result = self.run_audit(self.commit())
        self.assertEqual(result["status"], "potential_discrepancy")
        self.assertTrue(result["expert_confirmation_required"])
        self.assertEqual(result["evidence"]["paper"][0]["page"], 2)
        self.assertEqual(result["evidence"]["code"][0]["use"]["line_start"], 4)
        self.assertEqual(result["evidence"]["code"][0]["definitions"][0], {
            "path": "settings.py", "line_start": 2, "line_end": 2, "excerpt": "RATE = 0.05",
        })

    def test_equivalent_numeric_notation_at_requested_stage(self):
        self.article("Step 1: initial learning rate was 0.05, while Step 2: initial learning rate was 0.2.")
        self.write("fit.py", "LR = 5e-2\noptimizer = Adam(model.parameters(), lr=LR)\n")
        result = self.run_audit(self.commit())
        self.assertEqual((result["status"], result["paper_value"], result["code_value"]), ("match", "0.05", "0.05"))

    def test_missing_value_in_pdf(self):
        self.article("Step 1 used SGD. Step 2: initial learning rate was 0.2.")
        self.write("fit.py", "optimizer = SGD(model.parameters(), lr=0.05)\n")
        result = self.run_audit(self.commit())
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertIsNone(result["paper_value"])
        self.assertIn("PDF text has no explicit", result["explanation"])

    def test_missing_explicit_optimizer_value(self):
        self.article("Step 1: initial learning rate was 0.05.")
        self.write("fit.py", "optimizer = SGD(model.parameters())\n")
        result = self.run_audit(self.commit())
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertIsNone(result["code_value"])
        self.assertIn("no uniquely resolvable explicit optimizer", result["explanation"])

    def test_unlabelled_scripts_cannot_be_assigned_to_stages_by_matching_lr(self):
        self.article("Step 1: initial learning rate was 0.05. Step 2: initial learning rate was 0.2.")
        self.write("README.md", "python fit.py\npython refine.py\n")
        self.write("fit.py", "optimizer = SGD(model.parameters(), lr=0.05)\n")
        result = self.run_audit(self.commit())
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertIn("does not link the requested stage", result["explanation"])

    def test_ambiguous_rates_in_documented_script(self):
        self.article("Step 1: initial learning rate was 0.05.")
        self.write("fit.py", """optimizer = SGD([
    {'params': first.parameters(), 'lr': 0.05},
    {'params': second.parameters(), 'lr': 0.1},
])
""")
        result = self.run_audit(self.commit())
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertIn("multiple plausible", result["explanation"])

    def test_conditional_optimizer_leaves_value_unresolved(self):
        self.article("Step 1: initial learning rate was 0.05.")
        self.write("fit.py", """optimizer = SGD(model.parameters(), lr=0.05)
if fine_tune:
    optimizer = Adam(model.parameters(), lr=0.1)
""")
        result = self.run_audit(self.commit())
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertEqual(result["evidence"]["code"][0]["use"]["line_start"], 3)

    def test_launch_override_does_not_use_code_default_as_evidence(self):
        self.article("Step 1: initial learning rate was 0.05.")
        self.write("README.md", "Step 1: run `python fit.py --lr 0.1`.\n")
        self.write("fit.py", "optimizer = SGD(model.parameters(), lr=0.05)\n")
        result = self.run_audit(self.commit())
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertIn("command-line overrides", result["explanation"])

    def test_all_evidence_comes_from_selected_commit(self):
        self.article("Step 1: initial learning rate was 0.05.")
        self.write("settings.py", "class Settings:\n    RATE = 0.05\n\nsettings = Settings()\n")
        self.write("fit.py", "from settings import settings\noptimizer = Adam(params, lr=settings.RATE)\n"
                   "raise RuntimeError('Research code must never be executed')\n")
        pinned = self.commit()
        self.write("settings.py", "class Settings:\n    RATE = 0.2\n\nsettings = Settings()\n")
        newer = self.commit()
        self.write("README.md", "Step 1: python missing.py\n")
        result = self.run_audit(pinned[:12])
        self.assertEqual((result["status"], result["commit"], result["code_value"]), ("match", pinned, "0.05"))
        self.assertEqual(self.run_audit(newer)["code_value"], "0.2")
        self.assertEqual(self.git("rev-parse", "HEAD"), newer)
        self.assertEqual(self.git("status", "--short"), "M README.md")
