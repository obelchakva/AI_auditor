"""Command line entry point for static reproducibility checks."""

import argparse
import json
import re
from pathlib import Path

from .initial_lr import audit
from .rotation import audit_rotation


def main():
    parser = argparse.ArgumentParser(description="Compare a paper with a documented Git snapshot.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--group", help="Optimizer group, e.g. pretrained weights")
    parser.add_argument("--check", choices=("initial-lr", "test-rotation"), default="initial-lr")
    parser.add_argument("--dataset", help="Dataset named in the paper for test-rotation")
    parser.add_argument("--domain", help="Identifier of the test domain")
    parser.add_argument("--shell-recipe", nargs=3, metavar=("ARCH", "DATASET", "CLASSES_PER_WORKER"),
                        help="Select a documented shell recipe for initial-lr")
    parser.add_argument("--paper-section", help="Exact section heading, including its number when present")
    parser.add_argument("--paper-context", help="Text identifying the experiment in the initial LR statement")
    parser.add_argument("--paper-experiment", action="append", default=[], metavar="TEXT",
                        help="Require experiment text in a separate PDF paragraph (repeatable)")
    parser.add_argument("--paper-scope", help="Method named in both the experiment and default-LR section")
    parser.add_argument("--paper-model-pair", nargs=2, metavar=("TEACHER", "STUDENT"),
                        help="Verify a teacher/student pair in the paper's table")
    parser.add_argument("--table-shell-recipe", help="Pinned distributed shell recipe to inspect alongside the LR table")
    parser.add_argument("--paper-lr-table", nargs=3, metavar=("TABLE", "MODEL_METHOD", "DATASET"),
                        help="Select an LR cell by table number, exact group label and dataset column")
    parser.add_argument("--paper-table", type=int, help="Number of the table containing the selected model pair")
    parser.add_argument("--recipe-arg", action="append", default=[], metavar="FLAG=VALUE",
                        help="Require a literal README command argument, e.g. --recipe-arg=--model=small")
    parser.add_argument("--recipe-context", help="Text in the prose introducing the README command")
    parser.add_argument("--recipe-script", help="User-selected Python script on the pinned commit")
    parser.add_argument("--recipe-input", help="Script flag for the generated training input")
    parser.add_argument("--recipe-producer", help="Python script that writes the selected input")
    parser.add_argument("--recipe-transform", help="Local preparation function applied before writing the input")
    parser.add_argument("--paper-input", help="Dataset identifier shared by the PDF and generated input")
    parser.add_argument("--rotation-variant", help="Additional unlabelled domain of an in-memory test variant")
    parser.add_argument("--rotation-table", type=int, help="Paper table identifying the selected rotation variant")
    parser.add_argument("--rotation-producer", help="Pinned script producing the saved rotation configuration")
    parser.add_argument("--rotation-evaluator", help="Pinned script reading the rotation configuration for evaluation")
    parser.add_argument("--rotation-config-field", help="Configuration field holding the test domain")
    args = parser.parse_args()
    memory = (args.rotation_variant, args.rotation_table, args.rotation_producer,
              args.rotation_evaluator, args.rotation_config_field)
    if any(value is not None for value in memory) and (
        args.check != "test-rotation" or not all(memory) or args.rotation_table < 1
        or not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", args.rotation_variant)
        or not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", args.domain or "")
    ):
        parser.error("In-memory rotation requires all five rotation selectors and numeric test and variant domains")
    selected = (args.recipe_script, args.recipe_input, args.recipe_producer, args.recipe_transform, args.paper_input)
    if any(selected) and (not all(value and value.strip() for value in selected) or args.recipe_context
                          or not args.paper_section or not args.paper_context or args.paper_scope
                          or args.paper_experiment or args.paper_model_pair or args.paper_table):
        parser.error("A generated-input recipe requires script, input flag, producer, transform, paper input, "
                     "paper section and context, without README or other PDF selectors")
    paper = (args.paper_section, args.paper_context, args.paper_model_pair, args.paper_table,
             args.paper_experiment, args.paper_scope)
    recipe = (args.recipe_arg, args.recipe_context)
    table_lr = args.paper_lr_table
    if args.table_shell_recipe is not None and (not table_lr or not args.table_shell_recipe.strip()):
        parser.error("--table-shell-recipe requires --paper-lr-table and a nonempty script path")
    if table_lr and (not re.fullmatch(r"[1-9]\d{0,5}", table_lr[0])
                     or any(not value.strip() for value in table_lr[1:])
                     or any((*paper, *recipe, *selected))):
        parser.error("--paper-lr-table requires a positive table number and nonempty labels without other selectors")
    if any((*paper, *recipe, *selected, table_lr)) and (args.check != "initial-lr" or args.group or args.shell_recipe):
        parser.error("Explicit experiment selectors apply to initial-lr without --group or --shell-recipe")
    if any(paper) and not all(value and value.strip() for value in paper[:2]):
        parser.error("--paper-section and --paper-context are required together")
    if args.paper_table is not None and (args.paper_table < 1 or not args.paper_model_pair):
        parser.error("--paper-table requires a positive table number and --paper-model-pair")
    if bool(args.paper_experiment) != bool(args.paper_scope):
        parser.error("--paper-experiment and --paper-scope are required together")
    if args.paper_scope and (args.paper_model_pair or not args.paper_scope.strip()
                             or any(not term.strip() for term in args.paper_experiment)):
        parser.error("Cross-page selectors require nonempty terms and apply without --paper-model-pair")
    criteria = []
    for item in args.recipe_arg:
        flag, sep, value = item.partition("=")
        if not sep or not re.fullmatch(r"(?:--?[A-Za-z_][\w-]*|[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)", flag) or not value or any(key == flag for key, _ in criteria):
            parser.error("--recipe-arg requires a distinct literal FLAG=VALUE")
        criteria.append((flag, value))
    if any(selected) and (args.recipe_input not in dict(criteria) or any(
        not flag.startswith("-") or not re.fullmatch(r"[A-Za-z0-9_./+-]+", value) for flag, value in criteria
    )):
        parser.error("A generated-input recipe requires literal Python flags and the selected input argument")
    if not args.stage.strip():
        parser.error("--stage must contain a stage description")
    if args.shell_recipe:
        architecture, branch, count = args.shell_recipe
        if args.check != "initial-lr" or args.group:
            parser.error("--shell-recipe applies to initial-lr without --group")
        if not (architecture.isidentifier() and branch.isidentifier()
                and count.isascii() and count.isdecimal() and len(count) <= 7 and 0 < int(count) <= 1000000):
            parser.error("--shell-recipe requires a literal architecture, branch and positive class count")
    try:
        if args.check == "test-rotation":
            if not (args.dataset and args.dataset.strip() and args.domain and args.domain.strip()):
                parser.error("--check test-rotation requires --dataset and --domain")
            if args.group:
                parser.error("--group applies to the initial-lr check")
            result = audit_rotation(args.pdf, args.repo, args.commit, args.stage.strip(),
                                    args.dataset.strip(), args.domain.strip(), memory=memory if any(memory) else None)
        else:
            result = audit(args.pdf, args.repo, args.commit, args.stage.strip(), args.group, args.shell_recipe,
                           paper_selector=paper if any(paper) else None,
                           recipe_selector=(criteria, args.recipe_context) if any(recipe) and not any(selected) else None,
                           selected_script=(*selected, criteria) if any(selected) else None,
                           table_lr=(int(table_lr[0]), *table_lr[1:]) if table_lr else None,
                           table_shell_recipe=args.table_shell_recipe)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
