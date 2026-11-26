import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pymupdf


class TestRotation(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repo = Path(directory.name)
        self.pdf = self.repo / "article.pdf"
        self.git("init", "-q")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.invalid")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], text=True).strip()

    def fixture(self, *, angle=30, split="test", load_suffix="", competing=False):
        with pymupdf.open() as doc:
            page = doc.new_page()
            page.insert_textbox(pymupdf.Rect(50, 100, 540, 300),
                                "In the Meridian experiment on Tile Set, test domain east uses a rotation of 30 degrees.")
            doc.save(self.pdf)
        (self.repo / "README.md").write_text(
            "Generate Tile Set images:\n\n`python generate.py --dataset tiles`\n\n"
            "Evaluate Meridian on Tile Set:\n\n`python evaluate.py --dataset tiles --split test`\n",
            encoding="utf-8",
        )
        writer = """import argparse
import torch
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
args = parser.parse_args()

for domain in ['east', 'west']:
    split = {split!r}
    image = transforms.functional.rotate(pixels, {angle})
    path = f'data/{{args.dataset}}/{{split}}/{{domain}}.pt'
    torch.save(image, path)
""".format(split=split, angle=angle)
        if competing:
            writer += ("    other = transforms.functional.rotate(pixels, -30)\n"
                       "    torch.save(other, f'data/{args.dataset}/{split}/{domain}_second.pt')\n")
        (self.repo / "generate.py").write_text(writer, encoding="utf-8")
        evaluator = """import argparse
from reader import load_images

parser = argparse.ArgumentParser()
parser.add_argument('--dataset')
parser.add_argument('--split')
parser.add_argument('--test-domains', nargs='+', default=['east', 'west'])
args = parser.parse_args()
load_images(args.dataset, args.split, args.test_domains)
"""
        reader = """import torch

def load_images(dataset, split, domains):
    images = []
    for domain in domains:
        images.append(torch.load(f'data/{dataset}/{split}/{domain}SUFFIX.pt'))
"""
        reader = reader.replace("SUFFIX", load_suffix)
        if competing:
            reader += "        images.append(torch.load(f'data/{dataset}/{split}/{domain}_second.pt'))\n"
        (self.repo / "evaluate.py").write_text(evaluator, encoding="utf-8")
        (self.repo / "reader.py").write_text(reader, encoding="utf-8")
        self.git("add", "README.md", "generate.py", "evaluate.py", "reader.py")
        self.git("-c", "core.hooksPath=/dev/null", "commit", "-qm", "Freeze test recipe")
        return self.git("rev-parse", "HEAD")

    def audit(self, commit):
        result = subprocess.run([
            sys.executable, "-m", "ai_auditor", "--check", "test-rotation", "--pdf", str(self.pdf),
            "--repo", str(self.repo), "--commit", commit, "--stage", "Meridian",
            "--dataset", "Tile Set", "--domain", "east",
        ], check=True, capture_output=True, text=True)
        return json.loads(result.stdout)

    def memory_fixture(self, angle, use_rotated=True):
        with pymupdf.open() as doc:
            doc.new_page().insert_text((72, 72), "On Tile Set, we test on M75°.")
            page = doc.new_page()
            for row, line in enumerate((
                "In both setups M75° is the test domain. Unlabeled data from M30° is used.",
                "Table 2: Additional unlabeled data from M30°. Accuracy on M75°.",
                "Unsupervised domain M30°    80.2",
            )):
                page.insert_text((72, 72 + row * 35), line)
            doc.save(self.pdf)
        (self.repo / "prepare.py").write_text("""import argparse
import torch
from torch.utils import data

def train(loaders):
    for images in loaders["unsup"]:
        consume(images)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', default='75')
    args = parser.parse_args()
    args.domain = [args.domain]
    config_file = 'test_domain_' + str(args.domain[0]) + '.config'
    extra = data.DataLoader(Dataset(['30']))
    loaders = {}
    loaders["unsup"] = extra
    train(loaders)
    torch.save(args, config_file)
""", encoding="utf-8")
        (self.repo / "evaluate.py").write_text("""import torch
from torch.utils import data
from dataset import Images

def score(loader, classifier):
    for (x, y) in loader:
        classifier(x)

if __name__ == "__main__":
    args = torch.load('test_domain_75.config')
    loader = data.DataLoader(Images(args.domain, train=False))
    score(loader, classifier)
""", encoding="utf-8")
        returned = "changed" if use_rotated else "raw"
        (self.repo / "dataset.py").write_text(f"""from torchvision import datasets, transforms

class Images:
    def __init__(self, domain, train=True):
        self.domain = domain
        self.train = train
        if self.train:
            self.train_data, self.train_labels = self.prepare()
        else:
            self.test_data, self.test_labels = self.prepare()

    def prepare(self):
        if self.train:
            return training, labels
        raw = datasets.TileSet()
        angle = {angle}
        changed = transforms.functional.rotate(raw, angle)
        return {returned}, labels

    def __getitem__(self, index):
        if self.train:
            x = self.train_data[index]
        else:
            x = self.test_data[index]
        return x, label
""", encoding="utf-8")
        self.git("add", "prepare.py", "evaluate.py", "dataset.py")
        self.git("-c", "core.hooksPath=/dev/null", "commit", "-qm", "Freeze in-memory recipe")
        return self.git("rev-parse", "HEAD")

    def memory_audit(self, commit):
        result = subprocess.run([
            sys.executable, "-m", "ai_auditor", "--check", "test-rotation", "--pdf", str(self.pdf),
            "--repo", str(self.repo), "--commit", commit, "--stage", "Meridian", "--dataset", "Tile Set",
            "--domain", "75", "--rotation-variant", "30", "--rotation-table", "2",
            "--rotation-producer", "prepare.py", "--rotation-evaluator", "evaluate.py",
            "--rotation-config-field", "domain",
        ], check=True, capture_output=True, text=True)
        return json.loads(result.stdout)

    def test_memory_rotation_uses_actual_angle_not_configuration_name(self):
        result = self.memory_audit(self.memory_fixture("int(self.domain[0]) + -15"))
        self.assertEqual((result["status"], result["paper_value"], result["code_value"]),
                         ("potential_discrepancy", "75", "60"), result["explanation"])
        paper, = result["evidence"]["paper"]
        self.assertEqual((paper["page"], paper["variant_evidence"]["page"]), (1, 2))
        self.assertEqual(result["evidence"]["stage_links"][0]["configuration_pattern"], "test_domain_75.config")
        self.assertEqual(result["evidence"]["code"][0]["storage"], "in_memory")

    def test_memory_rotation_rejects_unrotated_pixels_sent_to_evaluation(self):
        result = self.memory_audit(self.memory_fixture("75", use_rotated=False))
        self.assertEqual((result["status"], result["paper_value"], result["code_value"]),
                         ("insufficient_materials", "75", None))
        self.assertIn("returned as test images", result["explanation"])
        self.assertIn("rotate", result["evidence"]["code"][0]["use"]["excerpt"])

    def test_memory_rotation_rejects_dynamic_angle(self):
        result = self.memory_audit(self.memory_fixture("choose_angle()"))
        self.assertEqual((result["status"], result["code_value"]), ("insufficient_materials", None))
        self.assertIn("rotate", result["evidence"]["code"][0]["use"]["excerpt"])

    def continued_domains(self, evaluation):
        with pymupdf.open() as doc:
            doc.new_page().insert_textbox(
                pymupdf.Rect(55, 72, 510, 145),
                "Rotated Tiles uses rotation to define each domain. Each of the images in the data",
            )
            doc.new_page().insert_textbox(
                pymupdf.Rect(55, 72, 510, 185),
                f"split when rotated by theta degrees is denoted Mtheta. {evaluation}",
            )
            doc.save(self.pdf)
        (self.repo / "README.md").write_text("Rotation tasks\n", encoding="utf-8")
        self.git("add", "README.md")
        self.git("-c", "core.hooksPath=/dev/null", "commit", "-qm", "Freeze PDF fixture")
        return subprocess.run([
            sys.executable, "-m", "ai_auditor", "--check", "test-rotation", "--pdf", str(self.pdf),
            "--repo", str(self.repo), "--commit", self.git("rev-parse", "HEAD"), "--stage", "Rotation tasks",
            "--dataset", "Rotated Tiles", "--domain", "90",
        ], check=True, capture_output=True, text=True)

    def test_continued_domain_definition_links_held_out_evaluation(self):
        result = json.loads(self.continued_domains("We evaluate on M0, M90. These are held-out domains.").stdout)
        self.assertEqual((result["paper_value"], result["code_value"]), ("90", None))
        paper, = result["evidence"]["paper"]
        self.assertEqual((paper["experiment_evidence"]["page"], paper["page"]), (1, 2))

    def test_continued_definition_does_not_use_training_or_other_domain_angle(self):
        result = json.loads(self.continued_domains(
            "We train on M90. We evaluate on M0, M60. These are held-out domains."
        ).stdout)
        self.assertEqual((result["status"], result["paper_value"]), ("insufficient_materials", None))
        self.assertEqual({item["page"] for item in result["evidence"]["paper"]}, {1, 2})

    def session_fixture(self, feed, angle="angle"):
        self.continued_domains("We evaluate on M0, M90. These are held-out domains.")
        (self.repo / "README.md").write_text(
            "# Rotation tasks\n\nExample:\npython launch.py --dataset tiles\n", encoding="utf-8")
        (self.repo / "launch.py").write_text("""import argparse
from model import Model

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--phase', default='train')
    args = parser.parse_args()
    return args

def main():
    args = parse()
    with tf.Session() as sess:
        model = Model(sess, args)
        if args.phase == 'train':
            model.test(full=True)

if __name__ == "__main__":
    main()
""", encoding="utf-8")
        (self.repo / "model.py").write_text(f"""from scipy.ndimage import rotate as rot

def rotate_image(image, angle):
    changed = rot(image, angle)
    return changed

class Model:
    def __init__(self, sess, args):
        self.sess = sess
        self.dataset_name = args.dataset
        if self.dataset_name == 'tiles':
            self.train_x, self.train_y, self.test_x, self.test_y = load_tiles()
        images = []
        angles = [0, 90]
        for angle in angles:
            for image in self.test_x:
                images.append(rotate_image(image, {angle}))
        self.full_test_x = np.array(images)

    def graph(self):
        return self.network(self.inputs, is_training=False)

    def test(self, full=False):
        if full:
            test_x = self.{feed}
        else:
            test_x = self.test_x
        for index in range(len(test_x)):
            batch = test_x[index:index + 1]
            feed = {{self.inputs: batch}}
            prediction = self.sess.run([self.test_accuracy], feed_dict=feed)
""", encoding="utf-8")
        self.git("add", "README.md", "launch.py", "model.py")
        self.git("-c", "core.hooksPath=/dev/null", "commit", "-qm", "Freeze batch evaluation")
        result = subprocess.run([
            sys.executable, "-m", "ai_auditor", "--check", "test-rotation", "--pdf", str(self.pdf),
            "--repo", str(self.repo), "--commit", self.git("rev-parse", "HEAD"), "--stage", "Rotation tasks",
            "--dataset", "Rotated Tiles", "--domain", "90",
        ], check=True, capture_output=True, text=True)
        return json.loads(result.stdout)

    def test_session_rotation_tracks_angle_used_by_scipy(self):
        result = self.session_fixture("full_test_x", "angle + 15")
        self.assertEqual((result["status"], result["paper_value"], result["code_value"]),
                         ("potential_discrepancy", "90", "105"), result["explanation"])
        self.assertIn("rot(image, angle)", result["evidence"]["code"][0]["use"]["excerpt"])

    def test_session_rotation_refuses_raw_images_in_the_feed(self):
        result = self.session_fixture("test_x")
        self.assertEqual((result["status"], result["paper_value"], result["code_value"]),
                         ("insufficient_materials", "90", None))
        self.assertIn("full-test evaluation", result["explanation"])
        self.assertTrue(any("feed_dict" in item["excerpt"] for item in
                            result["evidence"]["stage_links"][0]["stage_trace"]))

    def test_test_rotation_matches_on_frozen_commit(self):
        commit = self.fixture()
        (self.repo / "generate.py").write_text("raise RuntimeError('do not execute research code')\n")
        result = self.audit(commit)
        self.assertEqual((result["status"], result["paper_value"], result["code_value"]), ("match", "30", "30"))
        self.assertEqual(result["evidence"]["paper"][0]["page"], 1)
        code, = result["evidence"]["code"]
        self.assertEqual(code["artifact"], "data/tiles/test/east.pt")
        self.assertTrue(any(e["path"] == "reader.py" and "torch.load" in e["excerpt"] for e in code["stage_trace"]))

    def test_different_angles_for_same_named_test_domain(self):
        result = self.audit(self.fixture(angle=-30))
        self.assertEqual((result["status"], result["paper_value"], result["code_value"]),
                         ("potential_discrepancy", "30", "-30"))
        self.assertTrue(result["expert_confirmation_required"])

    def test_train_only_angle_remains_partial_evidence(self):
        result = self.audit(self.fixture(split="train"))
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertIsNone(result["code_value"])
        self.assertTrue(any(e.get("artifact") == "data/tiles/train/east.pt" for e in result["evidence"]["code"]))

    def test_rotation_cannot_be_linked_to_different_loaded_path(self):
        result = self.audit(self.fixture(load_suffix="_other"))
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertIn("could not be linked", result["explanation"])

    def test_competing_rotations_in_loaded_test_domain_are_ambiguous(self):
        result = self.audit(self.fixture(competing=True))
        self.assertEqual(result["status"], "insufficient_materials")
        self.assertIsNone(result["code_value"])
        self.assertEqual({e["value"] for e in result["evidence"]["code"]}, {"30", "-30"})
