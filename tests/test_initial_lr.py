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
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")

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

    def run_audit(self, commit, stage="Step 1", group=None, shell_recipe=None, extra=()):
        command = [
            sys.executable, "-m", "ai_auditor", "--pdf", str(self.pdf), "--repo", str(self.repo),
            "--commit", commit, "--stage", stage,
        ]
        if group:
            command.extend(("--group", group))
        if shell_recipe:
            command.extend(("--shell-recipe", *shell_recipe))
        command.extend(extra)
        process = subprocess.run(command, check=True, capture_output=True, text=True)
        return json.loads(process.stdout)

    def explicit_recipe(self, model="small", override="", update=""):
        self.article("2.1 Implementation details\n\n"
                     "For ImageBench, we initialize the learning rate as 0.05.")
        self.write("README.md", f"""Recipe without auxiliary loss:
```sh
python fit.py --model {model} -a 0 {override}
```
Recipe with auxiliary loss:
```sh
python fit.py --model {model} -a 1
```
""")
        self.write("fit.py", f"""import argparse
from torch import optim

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('-a', '--alpha', type=float)
    parser.add_argument('--learning_rate', type=float, default=0.05)
    args = parser.parse_args()
    if args.model in ['tiny']:
        args.learning_rate = 0.02
    return args

def main():
    cfg = parse()
{update}    optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate)

if __name__ == '__main__':
    main()
""")
        return self.commit(), ("--paper-section", "2.1 Implementation details",
                               "--paper-context", "ImageBench", f"--recipe-arg=--model={model}")

    def test_explicit_recipe_disambiguates_documented_commands(self):
        commit, args = self.explicit_recipe()
        result = self.run_audit(commit, extra=(*args, "--recipe-arg=-a=0",
                                              "--recipe-context", "without auxiliary loss"))
        self.assertEqual(result["status"], "match")
        self.assertEqual(result["paper_value"], "0.05")
        self.assertEqual(result["code_value"], "0.05")
        self.assertEqual(result["evidence"]["stage_links"][0]["line_start"], 3)

        ambiguous = self.run_audit(commit, extra=args)
        self.assertEqual(ambiguous["status"], "insufficient_materials")
        self.assertEqual(ambiguous["issue_codes"], ["recipe_ambiguous"])
        self.assertEqual(ambiguous["paper_value"], "0.05")
        self.assertIsNone(ambiguous["code_value"])
        self.assertEqual([r["line_start"] for r in ambiguous["evidence"]["stage_links"]], [3, 7])

    def test_explicit_recipe_honors_launch_override_and_code_precedence(self):
        for model, update, expected in (
            ("small", "", "0.07"), ("tiny", "", "0.02"),
            ("small", "    cfg.learning_rate = 0.03\n", "0.03"),
        ):
            with self.subTest(model=model, update=update):
                if self.pdf.exists():
                    self.pdf.unlink()
                commit, args = self.explicit_recipe(model, "--learning_rate=7e-2", update)
                result = self.run_audit(commit, extra=(*args, "--recipe-arg=-a=0"))
                self.assertEqual(result["status"], "potential_discrepancy")
                self.assertEqual(result["code_value"], expected)
                self.assertEqual(result["evidence"]["code"][0]["command_override"], "7e-2")

    def test_explicit_recipe_preserves_sites_when_code_override_is_unresolved(self):
        commit, args = self.explicit_recipe(update="    if unknown():\n        cfg.learning_rate = 0.03\n")
        result = self.run_audit(commit, extra=(*args, "--recipe-arg=-a=0"))
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertEqual(result["issue_codes"], ["code_lr_unresolved"])
        self.assertIsNone(result["code_value"])
        self.assertIn("optim.SGD", result["evidence"]["code"][0]["use"]["excerpt"])

    def hydra_fixture(self, experiment_method="PairFit", lr="5e-3", override=" lr=2e-2",
                      duplicate=False, appendix_heading="A PairFit training settings", extra_config=""):
        with pymupdf.open() as doc:
            doc.new_page().insert_text((72, 72), f"For paired dialogues we train Model-2 using {experiment_method}.")
            page = doc.new_page()
            page.insert_text((72, 72), appendix_heading)
            page.insert_textbox(pymupdf.Rect(72, 102, 440, 220),
                                "Unless noted otherwise, we use the RMSprop optimizer with a learning "
                                "rate of 0.02 by default. We warm up from 0 to 0.02.")
            doc.save(self.pdf)
        command = "python -u fit.py model=small datasets=[dialogue] loss=pair loss.beta=0.1 batch_size=64"
        self.write("README.md", command + override + "\n" + (command + " lr=0.03\n" if duplicate else ""))
        self.write("conf/train.yaml", f"""lr: {lr}
optimizer: RMSprop
trainer: Trainer
batch_size: 4
datasets: []
defaults:
  - _self_
  - model: small
  - loss: other
{extra_config}""")
        self.write("conf/model/small.yaml", "name_or_path: Example/Model-2\n")
        self.write("conf/loss/pair.yaml", "beta: 1\n")
        self.write("fit.py", """import hydra
import trainers

def worker(cfg):
    TrainerClass = getattr(trainers, cfg.trainer)
    selected = TrainerClass(model, cfg)
    selected.train()

@hydra.main(config_path="conf", config_name="train")
def main(cfg):
    worker(cfg)

if __name__ == "__main__":
    main()
""")
        self.write("trainers.py", """import torch

class Trainer:
    def __init__(self, model, cfg):
        self.config = cfg

    def train(self):
        optimizer = getattr(torch.optim, self.config.optimizer)(model.parameters(), lr=self.config.lr)
""")
        return self.commit(), ("--paper-section", appendix_heading,
                               "--paper-context", "RMSprop", "--paper-experiment", "paired dialogues",
                               "--paper-experiment", "Model-2", "--paper-scope", "PairFit",
                               "--recipe-arg=model=small", "--recipe-arg=datasets=[dialogue]", "--recipe-arg=loss=pair",
                               "--recipe-arg=loss.beta=0.1", "--recipe-arg=batch_size=64")

    def test_hydra_explicit_override_uses_base_rate_and_both_pages(self):
        commit, args = self.hydra_fixture()
        result = self.run_audit(commit, extra=args)
        self.assertEqual(result["status"], "match", result["explanation"])
        self.assertEqual((result["paper_value"], result["code_value"]), ("0.02", "0.02"))
        article = result["evidence"]["paper"][0]
        self.assertEqual((article["experiment_evidence"]["page"], article["page"]), (1, 2))
        code = result["evidence"]["code"][0]
        self.assertEqual(code["command_override"], "2e-2")
        self.assertEqual(code["definitions"][0]["excerpt"], "lr: 5e-3")
        self.assertIn("config.lr", code["use"]["excerpt"])

    def test_hydra_refuses_to_link_unrelated_pdf_context(self):
        commit, args = self.hydra_fixture(experiment_method="AnotherMethod")
        result = self.run_audit(commit, extra=args)
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertEqual(result["issue_codes"], ["paper_lr_unconfirmed"])
        self.assertIsNone(result["paper_value"])
        self.assertEqual(result["code_value"], "0.02")
        self.assertEqual({item["page"] for item in result["evidence"]["paper"]}, {1, 2})

    def test_hydra_preserves_ambiguous_readme_recipes(self):
        commit, args = self.hydra_fixture(duplicate=True)
        result = self.run_audit(commit, extra=args)
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertEqual(result["issue_codes"], ["recipe_ambiguous"])
        self.assertEqual([ref["line_start"] for ref in result["evidence"]["stage_links"]], [1, 2])
        self.assertEqual((result["paper_value"], result["code_value"]), ("0.02", None))

    def test_hydra_refuses_interpolated_lr_and_keeps_yaml_evidence(self):
        commit, args = self.hydra_fixture(lr="${base_lr}", override="", extra_config="base_lr: 0.02\n")
        result = self.run_audit(commit, extra=args)
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertEqual(result["issue_codes"], ["code_lr_unresolved"])
        self.assertIn("interpolated", result["explanation"])
        self.assertEqual(result["evidence"]["code"][0]["definitions"][0]["excerpt"], "lr: ${base_lr}")

    def generated_input_fixture(self, case="ALPHA", used=True):
        with pymupdf.open() as doc:
            page = doc.new_page()
            page.insert_text((72, 72), "Implementation details")
            page.insert_textbox(pymupdf.Rect(72, 95, 470, 165),
                                "The models are optimized using the Adam optimizer starting with a learning rate "
                                "of 0.03. With multiple renderings, we train for three epochs.")
            page.insert_textbox(pymupdf.Rect(72, 185, 470, 255),
                                "The ALPHA dataset is expanded with renderings before training.")
            doc.save(self.pdf)
        training = "data" if used else "other_data"
        self.write("fit.py", f"""import argparse
import pickle
import torch

def read(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def main(path):
    data = read(path)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = Trainer(model, {training}, optimizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    args = parser.parse_args()
    main(args.input)
""")
        self.write("prepare.py", """import pickle

def add_renderings(data):
    return [render(item) for item in data]

def main():
    alpha = load_alpha()
    beta = load_beta()
    alpha = add_renderings(alpha)
    beta = add_renderings(beta)
    with open('generated/train_ALPHA.pickle', 'wb') as out:
        pickle.dump(alpha, out)
    with open('generated/train_BETA.pickle', 'wb') as out:
        pickle.dump(beta, out)

if __name__ == '__main__':
    main()
""")
        return self.commit(), ("--paper-section", "Implementation details",
                               "--paper-context", "multiple renderings", "--paper-input", "ALPHA",
                               "--recipe-script", "fit.py", "--recipe-input=--input",
                               "--recipe-producer", "prepare.py", "--recipe-transform", "add_renderings",
                               f"--recipe-arg=--input=generated/train_{case}.pickle")

    def test_user_selected_script_checks_generated_input_before_comparing_lr(self):
        commit, args = self.generated_input_fixture()
        result = self.run_audit(commit, extra=args)
        self.assertEqual(result["status"], "potential_discrepancy", result["explanation"])
        self.assertEqual((result["paper_value"], result["code_value"]), ("0.03", "0.01"))
        self.assertEqual(result["evidence"]["stage_links"][0]["source"], "user_selected_script")
        self.assertIn("ALPHA", result["evidence"]["paper"][0]["input_evidence"]["excerpt"])
        trace = result["evidence"]["code"][0]["stage_trace"]
        self.assertTrue(any("pickle.dump(alpha" in item["excerpt"] for item in trace))
        self.assertFalse((self.repo / "generated/train_ALPHA.pickle").exists())

        other = self.run_audit(commit, extra=(*args[:-1], "--recipe-arg=--input=generated/train_BETA.pickle"))
        self.assertEqual(other["status"], "insufficient_materials")
        self.assertEqual(other["issue_codes"], ["code_lr_unresolved"])
        self.assertEqual(other["paper_value"], "0.03")
        self.assertIsNone(other["code_value"])
        self.assertIn("does not identify the dataset", other["explanation"])
        self.assertEqual(other["evidence"]["code"][0]["unconfirmed_value"], "0.01")

    def test_user_selected_script_refuses_input_read_but_unused_by_trainer(self):
        commit, args = self.generated_input_fixture(used=False)
        result = self.run_audit(commit, extra=args)
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertEqual(result["issue_codes"], ["code_lr_unresolved"])
        self.assertIsNone(result["code_value"])
        self.assertIn("training setup", result["explanation"])

    def transfer_appendix(self, rates):
        with pymupdf.open() as doc:
            page = doc.new_page()
            page.insert_text((72, 72), "B Experimental details")
            page.insert_textbox(
                pymupdf.Rect(72, 98, 488, 155),
                "We performed our hyperparameter search to optimize for Sample10 quality, "
                "then transferred the resulting settings to other datasets:",
            )
            for index, rate in enumerate(rates):
                page.insert_text((72, 170 + 24 * index), f"- We set the learning rate to {rate}")
            doc.save(self.pdf)
        return self.commit()

    def test_transfer_appendix_retains_paper_rate_when_recipe_is_unresolved(self):
        commit = self.transfer_appendix(["2e-4, and we lowered it to 2e-5 for large images."])
        audit = self.run_audit(commit, stage="Training details")
        self.assertEqual((audit["status"], audit["paper_value"], audit["code_value"]),
                         ("insufficient_materials", "0.0002", None))
        paper, = audit["evidence"]["paper"]
        self.assertEqual(paper["experiment_evidence"]["dataset"], "Sample10")
        self.assertIn("2e-4", paper["excerpt"])

    def test_transfer_appendix_refuses_lr_qualified_for_another_dataset(self):
        commit = self.transfer_appendix(["0.003 for Other20."])
        audit = self.run_audit(commit, stage="Training details")
        self.assertIsNone(audit["paper_value"])
        self.assertIn("paper_lr_unconfirmed", audit["issue_codes"])
        self.assertTrue(any("Other20" in item["excerpt"] for item in audit["evidence"]["paper"]))

    def test_transfer_appendix_preserves_conflicting_rates(self):
        commit = self.transfer_appendix(["0.0002.", "0.0003."])
        audit = self.run_audit(commit, stage="Training details")
        self.assertIsNone(audit["paper_value"])
        self.assertIn("paper_lr_ambiguous", audit["issue_codes"])
        self.assertEqual({item["value"] for item in audit["evidence"]["paper"]}, {"0.0002", "0.0003"})

    def fire_training_fixture(self, override="", duplicate=False, default="1e-3"):
        self.write("README.md", "The training and evaluation scripts are in the `jobs/` subdirectory.\n"
                   "The commands to run training and evaluation are in comments at the top of the scripts.\n")
        command = "python3 jobs/fit.py train --exp_name $NAME"
        extra = command + " --lr=0.04\n" if duplicate else ""
        self.write("jobs/fit.py", f'''"""Unconditional Sample10

{command}{override}
{extra}python3 jobs/fit.py evaluation --exp_name $NAME
"""
import fire
from core import worker, datasets

def train(exp_name, dataset='sample10', optimizer='adam', lr={default}, warmup=10):
    ds = datasets.get_dataset(dataset)
    worker.run_training(lr=lr, warmup=warmup, optimizer=optimizer, train_input_fn=ds.train_input_fn)

def evaluation(exp_name):
    pass

if __name__ == '__main__':
    fire.Fire()
''')
        self.write("core/datasets.py", """import functools
DATASETS = {"sample10": functools.partial(SimpleDataset, name="sample10")}

def get_dataset(name):
    return DATASETS[name]()
""")
        self.write("core/worker.py", """from . import utility

def run_training(*, lr, warmup, optimizer, train_input_fn):
    def callback(features):
        step = tf.train.get_or_create_global_step()
        warmed = utility.get_warmed_up_lr(max_lr=lr, warmup=warmup, global_step=step)
        optimizer_result = utility.make_optimizer(lr=warmed, optimizer=optimizer)
        return optimizer_result
    estimator = tf.estimator.tpu.TPUEstimator(model_fn=callback)
    estimator.train(input_fn=train_input_fn)
""")
        self.write("core/utility.py", """def get_warmed_up_lr(max_lr, warmup, global_step):
    if warmup == 0:
        return max_lr
    return max_lr * tf.minimum(tf.cast(global_step, tf.float32) / float(warmup), 1.0)

def make_optimizer(*, optimizer, lr):
    if optimizer == 'adam':
        result = tf.train.AdamOptimizer(learning_rate=lr)
    return result
""")
        return self.transfer_appendix(["2e-4."])

    def test_fire_explicit_lr_overrides_default_before_warmup(self):
        result = self.run_audit(self.fire_training_fixture(override=" --lr=2e-4"), stage="Training details")
        self.assertEqual(result["status"], "match", result["explanation"])
        self.assertEqual((result["paper_value"], result["code_value"]), ("0.0002", "0.0002"))
        code, = result["evidence"]["code"]
        self.assertEqual(code["base_lr_source"], "command_override")
        self.assertIn("--lr=2e-4", code["definitions"][0]["excerpt"])
        self.assertIn("tf.minimum", code["warmup_evidence"][1]["excerpt"])
        self.assertIn("AdamOptimizer", code["use"]["excerpt"])
        reference, = result["evidence"]["stage_links"]
        self.assertEqual(reference["subcommand"], "train")

    def test_fire_dynamic_lr_keeps_paper_evidence(self):
        result = self.run_audit(self.fire_training_fixture(override=" --lr=$RATE"), stage="Training details")
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertEqual((result["paper_value"], result["code_value"]), ("0.0002", None))
        self.assertIn("code_lr_unresolved", result["issue_codes"])
        self.assertIn("environment substitution", result["explanation"])

    def test_fire_ambiguous_training_commands_are_preserved(self):
        result = self.run_audit(self.fire_training_fixture(duplicate=True), stage="Training details")
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertEqual((result["paper_value"], result["code_value"]), ("0.0002", None))
        self.assertIn("recipe_ambiguous", result["issue_codes"])
        self.assertEqual(len(result["evidence"]["stage_links"]), 2)

    def ruled_hyperparameter_table(self, headers=("News", "Reviews"), duplicate_group=False, missing=False,
                                   conflicting=False):
        self.write("README.md", f"Table fixture {headers}, {duplicate_group}, {missing}, {conflicting}.\n")
        with pymupdf.open() as doc:
            page = doc.new_page()
            for y in (70, 97, 150, 203, 256):
                page.draw_line((62, y), (480, y))
            for x, label in zip((70, 175, 312, 412), ("Method", "Dataset", *headers)):
                page.insert_text((x, 88), label)
            for y, model, method, a, b in (
                (97, "Encoder small", "Adapter", "0.001", "0.002"),
                (150, "Encoder large", "Adapter", "0.003", "" if missing else "0.004"),
                (203, "Encoder large", "Adapter" if duplicate_group else "Full", "0.005", "0.006"),
            ):
                page.insert_text((70, y + 15), model)
                page.insert_text((70, y + 29), method)
                page.insert_text((175, y + 23), "Learning Rate")
                if a:
                    page.insert_text((312, y + 23), a)
                if b:
                    page.insert_text((412, y + 23), b)
                if conflicting and y == 150:
                    page.insert_text((175, y + 40), "Initial Learning Rate")
                    page.insert_text((412, y + 40), "0.009")
            page.insert_text((100, 276), "Table 3: Hyperparameters grouped by model and method.")
            doc.save(self.pdf)
        return self.commit()

    def test_table_rate_selects_model_method_and_dataset(self):
        result = self.run_audit(self.ruled_hyperparameter_table(), extra=(
            "--paper-lr-table", "3", "Encoder-large Adapter", "Reviews"))
        self.assertEqual((result["paper_value"], result["code_value"]), ("0.004", None), result["explanation"])
        item, = result["evidence"]["paper"]
        self.assertEqual((item["table"], item["dataset"], item["model_method"]), (3, "Reviews", "Encoder large Adapter"))
        self.assertEqual(item["cell"], "0.004")

    def test_table_rate_refuses_ambiguous_heading(self):
        for options, message in [
            ({"headers": ("Reviews", "Reviews")}, "column header is ambiguous"),
            ({"duplicate_group": True}, "model/method header is ambiguous"),
            ({"conflicting": True}, "conflicting LR rows"),
        ]:
            with self.subTest(options=options):
                if self.pdf.exists():
                    self.pdf.unlink()
                result = self.run_audit(self.ruled_hyperparameter_table(**options), extra=(
                    "--paper-lr-table", "3", "Encoder large Adapter", "Reviews"))
                self.assertIsNone(result["paper_value"])
                self.assertIn("paper_lr_ambiguous", result["issue_codes"])
                self.assertIn(message, result["explanation"])

    def test_table_rate_does_not_borrow_from_neighbor_cell(self):
        result = self.run_audit(self.ruled_hyperparameter_table(missing=True), extra=(
            "--paper-lr-table", "3", "Encoder large Adapter", "Reviews"))
        self.assertIsNone(result["paper_value"])
        self.assertTrue(any("0.003" in item["excerpt"] for item in result["evidence"]["paper"]))

    def test_distributed_shell_reads_continuations_and_process_count(self):
        from ai_auditor.distributed_lr import shell_launch
        from ai_auditor.initial_lr import Snapshot

        slash = chr(92)
        self.write("suite/train.sh", "\n".join([
            "export WORKERS=8",
            "python -m torch.distributed.launch --nproc_per_node=$WORKERS " + slash,
            "train.py --model_name_or_path encoder-large --task_name reviews " + slash,
            "--do_train --apply_lora --lora_r 8 --lora_alpha 16 " + slash,
            "--max_seq_length 128 --per_device_train_batch_size 4 --learning_rate 3e-4 " + slash,
            "--num_train_epochs 20 --warmup_ratio 0.06",
        ]))
        entry, count, args, sites = shell_launch(Snapshot(self.repo, self.commit()), "suite/train.sh")
        self.assertEqual((entry, count, args["learning_rate"]), ("suite/train.py", 8, "3e-4"))
        self.assertTrue(any("--learning_rate" in item["excerpt"] for item in sites))

    def test_distributed_recipe_batch_agreement_does_not_prove_variant(self):
        from ai_auditor.distributed_lr import comparison_issue

        paper = [{"model_method": "Encoder large LoRA", "dataset": "Reviews", "value": "0.0002",
                  "group_settings": {"Batch Size": {"value": "4"}}}]
        code = {"value": "0.0003", "arguments": {"model_name_or_path": "encoder-large", "task_name": "reviews"},
                "batch": {"per_device": 4, "effective": 32}}
        issue = comparison_issue(paper, code)
        self.assertIn("per-device or effective", issue)
        self.assertFalse(code["comparison"]["variant_confirmed"])
        self.assertEqual((paper[0]["value"], code["value"]), ("0.0002", "0.0003"))

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

    def test_adjacent_sentences_keep_competing_stages_separate(self):
        self.article(
            "Step 1 trains the encoder. The initial learning rate is 0.03. "
            "Step 2 trains the decoder. The initial learning rate is 0.2."
        )
        self.write("fit.py", "optimizer = SGD(params, lr=3e-2)\n")
        result = self.run_audit(self.commit())
        self.assertEqual(result["status"], "match")
        self.assertEqual([item["value"] for item in result["evidence"]["paper"]], ["0.03"])
        self.assertIn("Step 1 trains the encoder", result["evidence"]["paper"][0]["excerpt"])

    def test_module_entry_point_is_resolved_at_selected_commit(self):
        self.article("Step 1: initial learning rate was 0.03.")
        self.write("README.md", "Step 1: `python -m runner`\n")
        self.write("jobs/runner.py", "optimizer = Adam(params, lr=0.03)\n")
        pinned = self.commit()
        self.write("other/runner.py", "optimizer = Adam(params, lr=0.03)\n")
        ambiguous = self.commit()
        result = self.run_audit(pinned)
        self.assertEqual(result["status"], "match")
        self.assertEqual(result["evidence"]["stage_links"][0]["entry_point"], "jobs/runner.py")
        result = self.run_audit(ambiguous)
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertIn("multiple entry points", result["explanation"])
        self.assertEqual(result["evidence"]["stage_links"][0]["module_candidates"],
                         ["jobs/runner.py", "other/runner.py"])

    def test_module_option_links_paper_model_to_fit_and_literal(self):
        self.article(
            "ORBIT encoder.\nThe encoder is trained for 40 epochs using Adam "
            "with hyperparameters: learning rate = .03.\n"
            "Other encoder. The encoder uses an initial learning rate of 0.2."
        )
        self.write("README.md", """### ORBIT experiments

```shell
python -m orbit --task experiment_name
```

| experiment_name | detail |
| --- | --- |
| reconstruction | Train the reconstruction model |
| quick | Quick demonstration |
""")
        self.write("experiments/orbit.py", """from toolkit.models import EncoderOrbit
import argparse

def reconstruction(n_epochs=40):
    model = EncoderOrbit()
    model.fit(n_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    args = parser.parse_args()
    if args.task == "quick":
        reconstruction(n_epochs=1)
    elif args.task == "reconstruction":
        reconstruction()
""")
        self.write("src/toolkit/models.py", """class EncoderOrbit:
    def fit(self, epochs):
        self.rate = 3e-2
        optimizer = Adam(self.params, lr=self.rate)
""")
        result = self.run_audit(self.commit(), "ORBIT encoder")
        self.assertEqual(result["status"], "match")
        self.assertEqual([item["value"] for item in result["evidence"]["paper"]], ["0.03"])
        code, = result["evidence"]["code"]
        self.assertEqual(code["definitions"][0]["excerpt"], "self.rate = 3e-2")
        self.assertEqual(code["use"]["path"], "src/toolkit/models.py")
        self.assertIn("reconstruction", code["stage_trace"][0]["excerpt"])

    def test_dotted_module_can_use_package_main(self):
        self.article("Step 1: initial learning rate was 0.03.")
        self.write("README.md", "Step 1: `python -m suite.training`\n")
        self.write("src/suite/training/__main__.py", "optimizer = Adam(params, lr=0.03)\n")
        result = self.run_audit(self.commit())
        self.assertEqual(result["status"], "match")
        self.assertEqual(result["evidence"]["stage_links"][0]["entry_point"],
                         "src/suite/training/__main__.py")

    def grouped_fixture(self, *, labels="pre-trained layers and the classifier", boundary=False):
        with pymupdf.open() as doc:
            first = doc.new_page()
            first.insert_text((72, 72), "2.1. Training", fontname="hebo")
            first.insert_text((72, 110), "Image classifier.", fontname="hebo")
            first.insert_text((72, 131), "We train the image classifier using the")
            second = doc.new_page()
            if boundary:
                second.insert_text((72, 72), "2.2. Segmentation", fontname="hebo")
            second.insert_textbox(
                pymupdf.Rect(72, 110, 510, 300),
                f"SGD optimizer. Learning rates are initially set to .01 and .03 for the {labels}, respectively.",
            )
            doc.save(self.pdf)
        self.write("README.md", "Image classifier: `python fit.py --settings cfg/selected.yml`\n")
        self.write("cfg/selected.yml", "TRAIN:\n  RATE: 3e-2\n  HEAD_RATE: .01\n")
        self.write("cfg/other.yml", "TRAIN:\n  RATE: .01\n  HEAD_RATE: .03\n")
        self.write("model.py", """class Network:
    def __init__(self):
        self.encoder = Encoder()
        self.head = Head()
        self.scratch_layers = [self.head]

    def groups(self):
        groups = ([], [])
        for layer in self.modules():
            if layer in self.scratch_layers:
                groups[1].append(layer.weight)
            else:
                groups[0].append(layer.weight)
        return groups
""")
        self.write("fit.py", """import argparse
from model import Network

def train(settings):
    model = Network()
    model.encoder.load_state_dict(load(settings.MODEL.WEIGHTS))
    groups = model.groups()
    base = settings.TRAIN.RATE
    opt = SGD([{"params": groups[0], "lr": base},
               {"params": groups[1], "lr": settings.TRAIN.HEAD_RATE}])

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    settings = DEFAULT.clone()
    settings.merge_from_file(args.settings)
    train(settings)
""")

    def test_group_order_and_selected_yaml_are_proved_without_matching_values(self):
        self.grouped_fixture()
        commit = self.commit()
        result = self.run_audit(commit, "Image classifier", "pretrained weights")
        self.assertEqual((result["status"], result["paper_value"], result["code_value"]),
                         ("potential_discrepancy", "0.01", "0.03"))
        paper, = result["evidence"]["paper"]
        self.assertEqual((paper["stage_context"]["page"], paper["page"]), (1, 2))
        code, = result["evidence"]["code"]
        self.assertEqual(code["group_index"], 0)
        self.assertEqual(code["definitions"][0], {
            "path": "cfg/selected.yml", "line_start": 2, "line_end": 2, "excerpt": "RATE: 3e-2",
        })
        self.assertIn("merge_from_file(args.settings)", " ".join(e["excerpt"] for e in code["stage_trace"]))
        scratch = self.run_audit(commit, "Image classifier", "scratch weights")
        self.assertEqual((scratch["paper_value"], scratch["code_value"]), ("0.03", "0.01"))
        unspecified = self.run_audit(commit, "Image classifier")
        self.assertEqual(unspecified["status"], "insufficient_materials")
        self.assertIn("explicit optimizer group", unspecified["explanation"])

    def test_grouped_pdf_does_not_cross_a_new_section(self):
        self.grouped_fixture(boundary=True)
        result = self.run_audit(self.commit(), "Image classifier", "pretrained weights")
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertIsNone(result["paper_value"])

    def test_ambiguous_paper_group_stays_unresolved(self):
        self.grouped_fixture(labels="pre-trained layers and pre-trained weights")
        result = self.run_audit(self.commit(), "Image classifier", "pretrained weights")
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertIsNone(result["paper_value"])
        self.assertIn("does not uniquely identify", result["explanation"])

    def test_yaml_application_tags_are_rejected(self):
        self.grouped_fixture()
        self.write("cfg/selected.yml", "TRAIN:\n  RATE: !custom 3e-2\n")
        result = self.run_audit(self.commit(), "Image classifier", "pretrained weights")
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertIsNone(result["code_value"])
        self.assertIn("explicit tags or aliases", result["explanation"])

    def flag_recipe(self, *, extra="", mutation=""):
        self.article("North West. We train a classifier and report accuracy.")
        self.write("README.md", f"## Running\n\n`python fit.py --region north_west{extra}`\n")
        self.write("fit.py", """import tensorflow as tf
from network import Classifier

flags = tf.app.flags
OPTIONS = flags.FLAGS
flags.DEFINE_string('region', 'south', 'Training region')
flags.DEFINE_float('step_size', 0.04, 'Initial learning rate')
data = load_data(OPTIONS.region)
""" + mutation + """
factory = Classifier
model = factory(data)
raise RuntimeError("research code must not execute")
""")
        self.write("network.py", """import tensorflow as framework

settings = framework.app.flags
PARAMS = settings.FLAGS

class Classifier:
    def __init__(self, data):
        self.optimizer = framework.train.GradientDescentOptimizer(learning_rate=PARAMS.step_size)
""")
        return self.commit()

    def test_missing_paper_lr_preserves_verified_flag_recipe(self):
        result = self.run_audit(self.flag_recipe(), "North–West")
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertIsNone(result["paper_value"])
        self.assertEqual(result["code_value"], "0.04")
        self.assertIn("comparison is not possible", result["explanation"])
        self.assertIn("published code recipe at this commit", result["scope"])
        link, = result["evidence"]["stage_links"]
        self.assertEqual(link["excerpt"], "`python fit.py --region north_west`")
        code, = result["evidence"]["code"]
        self.assertEqual(code["definitions"][0]["line_start"], 7)
        self.assertEqual(code["use"]["path"], "network.py")
        self.assertEqual(code["use"]["line_start"], 8)
        self.assertTrue(any("load_data(OPTIONS.region)" in e["excerpt"] for e in code["stage_trace"]))

    def test_flag_recipe_rejects_documented_lr_override(self):
        result = self.run_audit(self.flag_recipe(extra=" --step_size=0.08"), "North–West")
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertIsNone(result["code_value"])
        self.assertIn("overrides the learning-rate flag", result["explanation"])

    def test_flag_recipe_rejects_mutation_through_namespace_alias(self):
        commit = self.flag_recipe(mutation="other = OPTIONS\nother.step_size = runtime_value\n")
        result = self.run_audit(commit, "North–West")
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertIsNone(result["code_value"])
        self.assertIn("can be modified by the training code", result["explanation"])

    def shell_recipe_fixture(self, second_rate=".04", second_dataset="tiles"):
        self.article("Standalone Relay. We report the final accuracy for the tile dataset.")
        self.write("README.md", """# Run

```bash
cd experiments
# To run standalone relay
sh recipes/run_<smallnet|largenet>.sh <dataset> <classes-per-worker>
```
""")
        prefix = "--clf smallnet --paradigm relay --optim sgd --non-iid $classes"
        self.write("experiments/recipes/run_smallnet.sh", f"""# The comments below are not commands.
tiles(){{
    # python fit.py --dataset tiles {prefix} --lr 99
    python fit.py --dataset tiles {prefix} --lr 4e-2 --threshold 0.1 &
    python fit.py --dataset {second_dataset} {prefix} --lr {second_rate} --threshold 0.2 &
}}
other(){{
    python fit.py --dataset other {prefix} --lr 0.5 --threshold 0.1 &
}}
classes=$2
$1
""")
        self.write("experiments/fit.py", """from setup import parse, Options
from worker import train

args = Options(parse())
for epoch in range(10):
    train(args, model)
raise RuntimeError("research code must not execute")
""")
        self.write("experiments/setup.py", """import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--clf', type=str)
    parser.add_argument('--paradigm', type=str, nargs='+')
    parser.add_argument('--optim', type=str)
    parser.add_argument('--non-iid', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--threshold', type=float)
    return parser.parse_args()

class Options:
    def __init__(self, args):
        self.dataset = args.dataset
        self.clf = args.clf
        self.paradigm = args.paradigm
        self.optim = args.optim
        self.non_iid = args.non_iid
        self.lr = args.lr
""")
        self.write("experiments/worker.py", """import torch.optim as opt

def train(args, model):
    worker(args, model)

def worker(args, model):
    return make_optimizer(args, model)

def make_optimizer(args, model):
    if args.optim == 'sgd':
        return opt.SGD(model.parameters(), lr=args.lr)
""")
        return self.commit()

    def test_shell_recipe_preserves_agreeing_rates_and_non_iid_evidence(self):
        result = self.run_audit(self.shell_recipe_fixture(), "Standalone Relay",
                                shell_recipe=("smallnet", "tiles", "2"))
        self.assertEqual((result["status"], result["paper_value"], result["code_value"]),
                         ("insufficient_materials", None, "0.04"))
        self.assertEqual(len(result["evidence"]["code"]), 2)
        for item in result["evidence"]["code"]:
            self.assertEqual(item["non_iid_count"], 2)
            self.assertEqual(item["use"]["path"], "experiments/worker.py")
            self.assertIn("lr=args.lr", item["use"]["excerpt"])
            self.assertTrue(any(s["excerpt"] == "classes=$2" for s in item["stage_trace"]))
        self.assertEqual(result["evidence"]["stage_links"][0]["path"], "README.md")

    def test_shell_recipe_with_conflicting_rates_keeps_candidates_but_no_value(self):
        result = self.run_audit(self.shell_recipe_fixture(second_rate=".08"), "Standalone Relay",
                                shell_recipe=("smallnet", "tiles", "2"))
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertIsNone(result["code_value"])
        self.assertEqual({e["value"] for e in result["evidence"]["code"]}, {"0.04", "0.08"})
        self.assertIn("conflicting initial learning rates", result["explanation"])

    def test_shell_branch_with_mixed_datasets_does_not_supply_rate(self):
        result = self.run_audit(self.shell_recipe_fixture(second_dataset="other"), "Standalone Relay",
                                shell_recipe=("smallnet", "tiles", "2"))
        self.assertIsNone(result["code_value"])
        self.assertIn("cannot be assigned to this SGD experiment", result["explanation"])
