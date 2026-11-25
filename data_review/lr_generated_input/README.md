# Начальный LR · скрипт с подготовленным входом

GeqShift используется как регрессионный кейс. Для arXiv:2311.12657v1 подтверждён SHA-256 `123ff3b03944ef32b6ad0840acb8b50311252026720d6b5e0d178b10ac79c14e`. Git-коммит `f96ff93ba0a2cc7509a5463e0897d9a4a556a798` разрешён как полный commit object. Версии CRD, DPO и SAM также сверены с [манифестом](../final_evaluation/cases.tsv). Исследовательские программы и генерация данных не запускались.

| Прогон | Статус | LR статьи / рецепта |
|---|---|---|
| [GeqShift с ¹³C](geqshift-selected.json) | `potential_discrepancy`, требуется подтверждение эксперта | `0.0003 / 0.0001` |
| [GeqShift с ¹H при сохранённом контексте ¹³C](geqshift-other-input.json) | `insufficient_materials`, `code_lr_unresolved` | `0.0003 / null` |
| [DPO, полный выбор](dpo-selected.json) | `potential_discrepancy` | `0.000001 / 0.0000005` |
| [DPO, неоднозначный выбор](dpo-ambiguous.json) | `insufficient_materials` | `0.000001 / null` |
| [CRD, полный выбор](crd-selected.json) | `match` | `0.05 / 0.05` |
| [CRD, неоднозначный выбор](crd-ambiguous.json) | `insufficient_materials` | `0.05 / null` |
| [SAM](sam-original.json) | `insufficient_materials` | `null / null` |

Все пять контрольных JSON DPO, CRD и SAM побайтно совпали с предыдущим прогоном. Исходный снимок CLI, манифест и ручная приёмка сверены по контрольным суммам с первым итоговым прогоном.

## Цепочка GeqShift

- PDF стр. 11, *Implementation details*. Один абзац задаёт Adam с начальным LR `3e-4` и уточняет режим для нескольких конформаций: три эпохи с последующим снижением LR с коэффициентом 0.1. Раздел о данных на той же странице связывает измерения ¹³C с набором, для которого генерируются молекулярные конформации.
- `main.py:200,207–208` объявляет и передаёт `--train_path`, строки 181, 149 и 137–138 ведут к чтению входа. Строки 143, 160 и 187 связывают прочитанный объект с обучением. `evidence.stage_links.source` содержит `user_selected_script`.
- `data/create_data.py:176,187,198,207–219` формирует углеродный обучающий набор и добавляет конформации. Строки 97–127 показывают подготовку, а 225–226 записывают `datasets/train_data_13C.pickle`. Сам файл данных для статической проверки не нужен.
- `main.py:185` задаёт и передаёт в Adam литерал `1e-4`. Строки 191 и 95 отдельно показывают последующее умножение LR на `0.1` после эпохи. Сравнение относится к выбранному скрипту с этим входом. Историческая команда авторов по этим материалам не установлена.

## Воспроизведение и ограничения

Из корня репозитория выполните `bash data_review/lr_generated_input/commands.sh`. Полные [команды](commands.sh) используют PDF и bare-репозитории в `/tmp/ai-final-eval`. Их можно восстановить по ссылкам, коммитам и SHA-256 из манифеста.

Для GeqShift указаны `--paper-section 'Implementation details'`, `--paper-context 'multiple conformations'`, `--paper-input 13C`, `--recipe-script main.py`, `--recipe-input=--train_path`, `--recipe-producer data/create_data.py`, `--recipe-transform add_conformations_to_train_dataset` и литеральный `--recipe-arg` с путём набора. Реализована статическая цепочка для литерального пути и простого argparse, локальной передачи аргумента, чтения и записи через pickle, локального преобразования данных и однозначного LR оптимизатора. Неоднозначная цепочка не даёт подтверждённого значения.

Изменены `ai_auditor/__main__.py` и `ai_auditor/initial_lr.py`, добавлен `ai_auditor/script_lr.py`. В `tests/test_initial_lr.py` добавлены проверки подготовленного входа, другого входа с тем же Adam и прочитанного, но не переданного тренеру входа.

`.venv/bin/python -m unittest discover -s tests` — **37 тестов прошли**. `git diff --check` и проверка синтаксиса `commands.sh` прошли.
