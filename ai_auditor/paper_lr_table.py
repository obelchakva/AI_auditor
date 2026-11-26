"""Resolve an LR cell from a ruled PDF table with grouped parameter rows."""

import re
from collections import defaultdict

from .experiment_lr import clean
from .initial_lr import NUMBER, compact, normalized, open_article


CAPTION = re.compile(r"^Table\s+(\d+)\s*[:.]", re.I)
RATE_LABELS = {"learning rate", "lr", "initial learning rate", "base learning rate"}


def group_key(label):
    # Preserve variant markers such as a dagger. Hyphens and spaces are
    # equivalent in printed model names.
    return clean(re.sub(r"(?<=\w)[-‐‑](?=\w)", " ", label))


def rows(words):
    result = []
    for word in sorted(words, key=lambda item: (item[1], item[0])):
        if not result or abs(word[1] - result[-1][0][1]) > 2:
            result.append([])
        result[-1].append(word)
    return [sorted(row, key=lambda word: word[0]) for row in result]


def phrase(words):
    return " ".join(word[4] for word in words)


def box(words):
    return [round(min(word[0] for word in words), 2), round(min(word[1] for word in words), 2),
            round(max(word[2] for word in words), 2), round(max(word[3] for word in words), 2)]


def ruled_regions(page, lower, caption):
    widths = defaultdict(set)
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] != "l":
                continue
            a, b = item[1:]
            if abs(a.y - b.y) <= 0.3 and lower < a.y < caption[1] and abs(a.x - b.x) > 60:
                widths[round(min(a.x, b.x), 1), round(max(a.x, b.x), 1)].add(round(a.y, 2))
    for (left, right), lines in widths.items():
        rules = sorted(lines)
        if len(rules) >= 3 and right >= caption[0] and left <= caption[2] and caption[1] - rules[-1] <= 36:
            yield left, right, rules


def center(word):
    return (word[0] + word[2]) / 2


def read_region(words, region, base, group, dataset, include_settings=False):
    left, right, rules = region
    bands = [[word for word in words if left <= center(word) <= right and top < (word[1] + word[3]) / 2 < bottom]
             for top, bottom in zip(rules, rules[1:])]
    headers = [(i, row) for i, band in enumerate(bands) for row in rows(band)
               if len(row) >= 3 and clean(row[0][4]) in {"method", "model", "model/method"}
               and clean(row[1][4]) in {"dataset", "parameter", "setting"}]
    if len(headers) != 1:
        return [], [], "Cannot identify one model/method and dataset header row."
    header_index, header = headers[0]
    columns = header[2:]
    selections = [i for i, word in enumerate(columns) if clean(word[4]) == clean(dataset)]
    header_evidence = {**base, "excerpt": phrase(header), "column_headers": [word[4] for word in columns],
                       "header_bbox": box(header)}
    if len(selections) != 1:
        reason = "The dataset column header is ambiguous." if selections else "The dataset column is absent."
        return [], [header_evidence], reason
    selected = selections[0]
    centers = [center(word) for word in columns]
    if len(centers) > 1 and any(a >= b for a, b in zip(centers, centers[1:])):
        return [], [header_evidence], "Dataset columns overlap."
    parameter_left = header[1][0] - 2
    parameter_right = (center(header[1]) + centers[0]) / 2
    # Parameter labels may extend rightward beyond the center gap of the
    # narrowest header. Keep them apart from the first data column.
    parameter_right = max(parameter_right, header[1][2], columns[0][0] - 2)
    matching_groups, hits, partial = 0, [], [header_evidence]
    for band in bands[header_index + 1:]:
        group_words = [word for row in rows(band) for word in row if center(word) < parameter_left]
        if not group_words or group_key(phrase(group_words)) != group_key(group):
            continue
        matching_groups += 1
        parameter_rows = rows([word for word in band if center(word) >= parameter_left])
        boundaries = [parameter_right, *((a + b) / 2 for a, b in zip(centers, centers[1:])), right]
        settings = group_settings(parameter_rows, parameter_right, boundaries, centers, selected) if include_settings else {}
        for line in parameter_rows:
            parameters = [word for word in line if center(word) < parameter_right]
            if clean(phrase(parameters)) not in RATE_LABELS:
                continue
            item = {**header_evidence, "excerpt": phrase(line), "parameter": phrase(parameters),
                    "model_method": phrase(group_words), "group_bbox": box(group_words),
                    "dataset": columns[selected][4], "column_bbox": box([columns[selected]]), "row_bbox": box(line)}
            partial.append(item)
            cells = [word for word in line if boundaries[selected] <= center(word) < boundaries[selected + 1]]
            if len(cells) != 1 or not re.fullmatch(NUMBER, cells[0][4]):
                continue
            value = normalized(cells[0][4])
            if value is None:
                continue
            width = boundaries[selected + 1] - boundaries[selected]
            if (abs(center(cells[0]) - centers[selected]) <= width * 0.3
                    and cells[0][0] >= boundaries[selected] and cells[0][2] <= boundaries[selected + 1]):
                hits.append({**item, "value": value, "cell": cells[0][4], "cell_bbox": box(cells), **({"group_settings": settings} if include_settings else {})})
    if matching_groups != 1:
        return [], [*partial, *hits], ("The model/method header is ambiguous." if matching_groups
                                       else "The model/method group is absent.")
    if len(partial) > 2 or len(hits) > 1:
        return [], [*partial, *hits], "The selected group has ambiguous or conflicting LR rows."
    if len(hits) != 1:
        return [], partial, "The selected group has no numeric LR cell for the dataset."
    return hits, partial, None


def group_settings(lines, left, boundaries, centers, selected):
    settings = {}
    for line in lines:
        label = phrase([word for word in line if center(word) < left])
        if not label or clean(label) in RATE_LABELS:
            continue
        values = [word for word in line if center(word) >= left]
        cells = [word for word in values if boundaries[selected] <= center(word) < boundaries[selected + 1]]
        shared = (len(values) == 1 and abs(center(values[0]) - (left + boundaries[-1]) / 2) <= 13
                  and all(abs(center(values[0]) - c) > 5 for c in centers))
        selected_words = values if shared or re.search(r"config\.", clean(label)) else cells
        if selected_words:
            settings[label] = {"value": phrase(selected_words), "row": phrase(line),
                               "scope": "shared" if shared else "row" if re.search(r"config\.", clean(label)) else "column"}
    return settings


def variant_note(article, table, group):
    marker = re.search(r"[†‡]$", group)
    if not marker:
        return []
    return [{"page": page.number + 1, "excerpt": compact(block[4])}
            for page in article for block in page.get_text("blocks")
            if block[6] == 0 and re.search(rf"marked with\s*{marker[0]}", compact(block[4]), re.I)
            and re.search(rf"\bTable\s+{table}\b", compact(block[4]), re.I)]


def table_rate(pdf, table, group, dataset, *, include_settings=False):
    captions = []
    with open_article(pdf) as article:
        notes = variant_note(article, table, group) if include_settings else []
        for page in article:
            blocks = sorted((b for b in page.get_text("blocks") if b[6] == 0 and CAPTION.match(compact(b[4]))),
                            key=lambda b: b[1])
            for index, caption in enumerate(blocks):
                text = compact(caption[4])
                if int(CAPTION.match(text)[1]) != table:
                    continue
                base = {"page": page.number + 1, "table": table, "caption": text, "excerpt": text}
                words = page.get_text("words")
                lower = blocks[index - 1][3] if index else 0
                regions = list(ruled_regions(page, lower, caption))
                if len(regions) != 1:
                    captions.append(([], [base], "The caption does not identify a unique ruled table."))
                    continue
                captions.append(read_region(words, regions[0], base, group, dataset, include_settings))
    if len(captions) != 1:
        partial = [item for hits, evidence, _ in captions for item in [*evidence, *hits]]
        return [], partial, "The requested table caption is absent or ambiguous."
    hits, partial, issue = captions[0]
    if notes:
        for item in hits:
            item["variant_evidence"] = notes
    return hits, partial, issue
