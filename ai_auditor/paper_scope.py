"""Link an explicitly named experiment to a method-wide default base rate."""

import re

from .experiment_lr import INITIALIZE, clean
from .initial_lr import NUMBER, PAPER_LR, compact, normalized, open_article

DEFAULT_LR = re.compile(
    rf"\blearning rate\s+of\s+({NUMBER})(?![\w/%+]|\.\d)\s+by default\b", re.I
)
BASE_LR = re.compile(rf"\b(?:learning rate\s+of|(?:base|initial)\s+(?:lr|learning rate)\s*(?:=|is|of))\s+({NUMBER})"
                     r"(?![\w/%+]|\.\d)", re.I)
SECTION = re.compile(r"^(?:[A-Z](?:\.\d+)*|\d+(?:\.\d+)*)\s+[A-Z][^.!?]{2,100}$")


def term(text):
    return re.compile(rf"(?<!\w){re.escape(clean(text))}(?!\w)")


def scoped_default(pdf, section, optimizer, experiment_terms, scope):
    """Require a shared method and an explicit default applying across its experiments."""
    experiment_patterns = [term(value) for value in experiment_terms]
    scope_pattern = term(scope)
    optimizer_pattern = term(optimizer)
    experiments, candidates, partial, competing = [], [], [], []
    active = False
    headings = []
    with open_article(pdf) as article:
        for page in article:
            for block in page.get_text("blocks"):
                if block[6] != 0 or not block[4].strip():
                    continue
                text = compact(block[4])
                clean_text = clean(text)
                item = {"page": page.number + 1, "excerpt": text}
                if all(pattern.search(clean_text) for pattern in experiment_patterns):
                    if scope_pattern.search(clean_text):
                        if item not in experiments:
                            experiments.append(item)
                    if item not in partial:
                        partial.append(item)
                if clean_text == clean(section):
                    headings.append(item)
                    active = True
                    partial.append(item)
                    continue
                if SECTION.fullmatch(text) and len(block[4].strip().splitlines()) <= 2:
                    active = False
                if not active:
                    continue
                for sentence in re.split(r"(?<=[.!?])\s+", text):
                    default = DEFAULT_LR.search(sentence)
                    if default and optimizer_pattern.search(clean(sentence)):
                        partial.append(item)
                        if re.match(r"unless (?:noted|stated|specified) otherwise\b", sentence, re.I):
                            value = normalized(default[1])
                            if value is not None:
                                candidates.append({**item, "value": value})
                    for pattern in (BASE_LR, PAPER_LR, INITIALIZE):
                        for match in pattern.finditer(sentence):
                            value = normalized(match[1])
                            if value is not None:
                                competing.append(value)
                                if item not in partial:
                                    partial.append(item)
    evidence = partial + [item for item in experiments if item not in partial]
    if (len(headings) != 1 or not scope_pattern.search(clean(section)) or len(experiments) != 1
            or len(candidates) != 1 or any(value != candidates[0]["value"] for value in competing)
            or any(pattern.search(experiments[0]["excerpt"])
                   for pattern in (BASE_LR, PAPER_LR, INITIALIZE))):
        return [], evidence
    return [{**candidates[0], "section": section, "section_evidence": headings[0],
             "experiment_evidence": experiments[0], "optimizer": optimizer,
             "scope_link": "The method in the experiment paragraph is named in the section heading, "
                           "and the section states a default unless noted otherwise."}], evidence


def text_blocks(article):
    """Yield PDF text with source page and block boundaries intact."""
    for page in article:
        for block in page.get_text("blocks"):
            if block[6] == 0 and block[4].strip():
                text = re.sub(r"(?<=[^\W\d_])-\s*\n(?=[^\W\d_])", "", block[4])
                yield {"page": page.number + 1, "excerpt": compact(text)}


SCIENTIFIC = rf"(?:\d+(?:\.\d+)?\s*[×·]\s*10\s*[−-]?\s*\d+|{NUMBER})"
SET_RATE = re.compile(rf"\bset the learning rate to\s+({SCIENTIFIC})(?![\w%×]|\.\d|\s*[×·])", re.I)
TRANSFER_SCOPE = re.compile(
    r"\bhyperparameter (?:search|selection|tuning)\b[^.!?]*?\b(?:for|on)\s+"
    r"(?P<dataset>[A-Z][\w-]*)\b[^.!?]*?\bthen transferred\b[^.!?]*?\bto (?:the )?other datasets\s*:",
)


def numeric_rate(text):
    scientific = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[×·]\s*10\s*([−-]?)\s*(\d+)", text)
    if scientific:
        sign = "-" if scientific[2] else ""
        return normalized(f"{scientific[1]}e{sign}{scientific[3]}")
    return normalized(text)


def transferred_training_rate(pdf, stage):
    """Read a source dataset's baseline before explicitly scoped transfers."""
    if not re.fullmatch(r"(?:training|experimental) details", clean(stage)):
        return [], []
    experiment, section, candidates, partial = None, None, [], []
    with open_article(pdf) as article:
        for item in text_blocks(article):
            text = item["excerpt"]
            if SECTION.fullmatch(text):
                section = item if re.fullmatch(r"[A-Z](?:\.\d+)* (?:Experimental|Training) details", text, re.I) else None
                experiment = None
                continue
            if section is None:
                continue
            match = TRANSFER_SCOPE.search(text)
            if match:
                experiment = {**item, "dataset": match["dataset"]}
                partial.append(experiment)
                continue
            if experiment is None:
                continue
            if not re.match(r"^(?:•|-)\s", text):
                experiment = None
                continue
            for bullet in re.split(r"(?:^|\s)(?:•|-)\s+", text)[1:]:
                for rate in SET_RATE.finditer(bullet):
                    citation = {**item, "excerpt": bullet.strip(), "experiment_evidence": experiment,
                                "section_evidence": section}
                    suffix = re.split(r"[.!?](?=\s|$)", bullet[rate.end():], 1)[0]
                    # Allow a plain base-rate statement or a following explicit
                    # exception. Do not inherit a rate qualified by another dataset.
                    if not re.fullmatch(
                        rf"\s*(?:without any sweeping)?\s*(?:,\s*and\s+we\s+(?:lowered|raised|changed) "
                        rf"it to\s+{SCIENTIFIC}\s+for\s+[^,]+(?:,.*)?)?\s*", suffix, re.I
                    ):
                        partial.append(citation)
                        continue
                    value = numeric_rate(rate[1])
                    if value is not None:
                        candidates.append({**citation, "value": value,
                                           "scope_kind": "dataset_transfer",
                                           "scope_link": "The appendix names the source dataset for the settings "
                                                         "and separates the base rate from transfer exceptions."})
    if len({clean(item["experiment_evidence"]["dataset"]) for item in candidates}) > 1:
        return [], partial + candidates
    return candidates, partial


def evaluated_rotation_domains(pdf, dataset, stage, domain):
    """Read a continued definition mapping a domain label to its rotation."""
    if clean(stage) not in {"rotation tasks", "rotation task"} or not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", domain):
        return [], []
    dataset_pattern = term(dataset)
    candidates, partial = [], []
    with open_article(pdf) as article:
        blocks = list(text_blocks(article))
    for i, left in enumerate(blocks[:-1]):
        text = left["excerpt"]
        if not dataset_pattern.search(clean(text)) or not re.search(r"\b(?:rotation|rotated)\b", text, re.I):
            continue
        # Require the open clause to continue grammatically into the domain
        # definition. Headers and footnotes alone cannot establish a link.
        if not re.search(r"\b(?:each|all) of the images in the data$", text, re.I):
            continue
        for right in blocks[i + 1:]:
            if right["page"] > left["page"] + 1:
                break
            claim = right["excerpt"]
            rule = re.match(
                r"split when rotated by (?P<symbol>θ|Theta|theta) degrees is denoted (?P<prefix>[A-Z])(?P=symbol)\.", claim
            )
            if not rule:
                continue
            partial.extend([left, right])
            prefix = re.escape(rule["prefix"])
            label = rf"{prefix}([+-]?\d+(?:\.\d+)?)"
            for evaluation in re.finditer(rf"\bwe evaluate on\s+({label}(?:\s*(?:,|and)\s*{label})*)\s*\.", claim, re.I):
                if not re.search(r"\b(?:outside the span of train domains|held.out|unseen|test domains?)\b",
                                 claim[evaluation.end():], re.I):
                    continue
                from .rotation import angle_text

                listed = {angle_text(item) for item in re.findall(label, evaluation[1], re.I)}
                if angle_text(domain) in listed:
                    candidates.append({**right, "value": angle_text(domain), "experiment_evidence": left,
                                       "scope_link": "The continued definition maps the domain label to degrees "
                                                     "and explicitly lists it for held-out evaluation."})
    return candidates, partial
