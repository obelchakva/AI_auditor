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
