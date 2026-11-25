# Начальный LR · явный выбор Hydra-рецепта

DPO используется как регрессионный кейс. SHA-256 всех трёх PDF и полные Git-коммиты DPO, CRD и SAM сверены с [манифестом](../final_evaluation/cases.tsv). Исследовательский код и Hydra не запускались.

| Прогон | Статус | LR статьи / рецепта | Доказательства |
|---|---|---|---|
| [DPO, полный выбор](dpo-selected.json) | `potential_discrepancy` | `0.000001 / 0.0000005` | PDF стр. 9 и 20, README:85, `config/config.yaml:51`, `trainers.py:276–277`. Требуется подтверждение эксперта. |
| [DPO, без выбора loss и beta](dpo-ambiguous.json) | `insufficient_materials` | `0.000001 / null` | `recipe_ambiguous`. Сохранены команды SFT и DPO из README:75 и README:85. |
| [CRD, полный выбор](crd-selected.json) | `match` | `0.05 / 0.05` | README:59. |
| [CRD, неоднозначный выбор](crd-ambiguous.json) | `insufficient_materials` | `0.05 / null` | `recipe_ambiguous`. README:59 и README:64. |
| [SAM, исходная команда](sam-original.json) | `insufficient_materials` | `null / null` | `paper_lr_unconfirmed` и `recipe_not_found`. |

## Доказательства DPO

- PDF стр. 9 связывает однопроходный диалог Anthropic HH и Pythia-2.8B с обучением DPO после Preferred-FT. Приложение B озаглавлено *DPO Implementation Details and Hyperparameters*. На стр. 20 в нём указан LR RMSprop по умолчанию `1e-6`, с оговоркой об исключениях. Приведённое исключение для TL;DR меняет β. Статья отдельно описывает линейный прогрев.
- README:85 задаёт `model=pythia28`, `datasets=[hh]`, `loss=dpo`, `loss.beta=0.1`, `batch_size=64` и `trainer=FSDPTrainer`. В `config/model/pythia28.yaml:1` этому варианту соответствует `EleutherAI/pythia-2.8b`.
- `train.py:48` фиксирует каталог и имя конфигурации. `config/config.yaml:51` даёт `5e-7`, строка 80 выбирает RMSprop. В команде нет переопределения `lr`. `train.py:40–44,111–114` передаёт конфигурацию воркеру и выбранному тренеру. `FSDPTrainer` наследует `BasicTrainer`. В `trainers.py:276` конструктор оптимизатора получает `self.config.lr`, а прогрев задаётся отдельно на строке 277.

## Селекторы и границы поддержки

Для связи PDF указаны `--paper-experiment 'single-turn dialogue'`, `--paper-experiment 'Anthropic HH'`, `--paper-experiment 'Pythia-2.8B'`, `--paper-scope DPO`, точное название приложения B в `--paper-section` и `--paper-context RMSprop`. Свидетельства должны содержать общий метод и явный LR по умолчанию с оговоркой об исключениях. При отсутствии однозначной связи сохраняются частичные фрагменты.

Поддерживаются однострочные Python-команды с `-u`, литеральные `key=value` и плоские списки, локальный `@hydra.main`, корневой YAML и одноуровневые именованные группы из `defaults`. Явный численный `lr=` имеет приоритет. Вложенные `defaults`, директивы пакетов, пользовательские YAML-теги, интерполяция LR и динамическое изменение отслеживаемых настроек приводят к отказу. Интерполяция сторонних параметров вроде пути журнала не вычисляется. Анализатор сравнивает базовый LR при создании оптимизатора. Расписание прогрева сохраняется как контекст.

## Воспроизведение

Из корня репозитория запустите `bash data_review/lr_hydra_experiment/commands.sh`. [Команды](commands.sh) используют локальные PDF и bare-репозитории в `/tmp/ai-final-eval`. После очистки каталога их можно восстановить по `pdf_url`, `pdf_sha256`, `repo_url` и `commit` из манифеста.

Обновлены `ai_auditor/__main__.py`, `initial_lr.py` и `experiment_lr.py`. Добавлены `paper_scope.py` и `hydra_lr.py`. В `tests/test_initial_lr.py` добавлены проверки переопределения LR, неподтверждённой связи PDF, неоднозначной команды и отказа при интерполяции LR.

`.venv/bin/python -m unittest discover -s tests` — **35 тестов прошли**. `git diff --check` прошёл. Исходные прогоны и эталонные материалы сохранены.
