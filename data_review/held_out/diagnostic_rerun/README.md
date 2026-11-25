# Диагностический повтор BP Benchmark и FedOptim

25.11.2026. Оба запуска завершились с кодом 0. Команды, включая пути,
коммиты и `--stage`, взяты из метаданных первого прогона без изменений.
Рабочий каталог текущего прогона — корень проекта, поэтому
`python -m ai_auditor` использовал текущие исходники. SHA-256 каждого
PDF побайтно совпал с сохранённым в первом прогоне.

| Кейс | Поле | До | После |
| --- | --- | --- | --- |
| BP Benchmark `183bd763` | `status` | `insufficient_materials` | `insufficient_materials` |
| BP Benchmark `183bd763` | `paper_value` / `code_value` | `null` / `null` | `null` / `null` |
| BP Benchmark `183bd763` | `evidence.stage_links` / `evidence.code` | `[]` / `[]` | `[]` / `[]` |
| BP Benchmark `183bd763` | Причина | В PDF отсутствует явный LR для этапа. README не связан с Python-точкой входа. | Те же причины. Появилось уточнение, что сравнение невозможно. |
| FedOptim `b55880a0` | `status` | `insufficient_materials` | `insufficient_materials` |
| FedOptim `b55880a0` | `paper_value` / `code_value` | `null` / `null` | `null` / `null` |
| FedOptim `b55880a0` | `evidence.stage_links` / `evidence.code` | `[]` / `[]` | `[]` / `[]` |
| FedOptim `b55880a0` | Причина | В PDF отсутствует явный LR для этапа. README не связан с Python-точкой входа. | Те же причины. Появилось уточнение, что сравнение невозможно. |

Для обоих кейсов единственное изменение JSON — дополнение в `explanation`:
`so a comparison is not possible`.

## BP Benchmark

**Первое неподтверждённое звено — выбор команды.** CLI не связал
`--stage 'Sig2Lab ResNet1D training'` с заголовком `Training Sig2Lab models`
и командой в README. В результате CLI не показал `lr: 0.001` и не учёл
`auto_lr_find: true`. [README:227–235](https://github.com/inventec-ai-center/bp-benchmark/blob/813e1a6a938e2698d348129f2bcc01de44450df9/README.md#L227-L235)

При ручном чтении видно, что выбранный YAML содержит `param_model.lr: 0.001`
и `param_trainer.auto_lr_find: true`. Код передаёт настройки в `MyTrainer`,
а `configure_optimizers` передаёт `param_model.lr` в Adam. `0.001` можно
сохранить как частичное доказательство значения в конфигурации. Значение
после подбора по доступным фрагментам установить нельзя.
[YAML:22–26](https://github.com/inventec-ai-center/bp-benchmark/blob/813e1a6a938e2698d348129f2bcc01de44450df9/code/train/core/config/dl/resnet/resnet_bcg.yaml#L22-L26),
[YAML:33–44](https://github.com/inventec-ai-center/bp-benchmark/blob/813e1a6a938e2698d348129f2bcc01de44450df9/code/train/core/config/dl/resnet/resnet_bcg.yaml#L33-L44),
[solver_s2l.py:122–133](https://github.com/inventec-ai-center/bp-benchmark/blob/813e1a6a938e2698d348129f2bcc01de44450df9/code/train/core/solver_s2l.py#L122-L133),
[base_pl.py:83–89](https://github.com/inventec-ai-center/bp-benchmark/blob/813e1a6a938e2698d348129f2bcc01de44450df9/code/train/core/models/base_pl.py#L83-L89)

[Старый JSON](../first_run/183bd763.json) · [Новый JSON](183bd763.json) ·
[Команда и SHA-256 PDF](183bd763.run.json).

## FedOptim

**Первое неподтверждённое звено — выбор команды.** Для этапа
`Standalone LBGM` README показывает шаблон shell-команды с выбором
архитектуры и датасета. CLI не дошёл до ветки `cifar()` скрипта CNN и не
связал `--optim sgd --lr 1e-2` с локальным оптимизатором.
[README:38–39](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/README.md#L38-L39)

При ручном чтении команды в ветке `cifar()` с `--non-iid $n` используют
`--optim sgd --lr 1e-2` при разных порогах LBGM. Парсер принимает
`--lr` как число, а локальный worker вызывает `get_optim(args, node_model)`.
Ветка SGD передаёт `args.lr` в оптимизатор. Это значение опубликованной
команды. [Скрипт:25–32](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/src/sh/train_lbgm_cnn.sh#L25-L32),
[argparser.py:50](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/src/common/argparser.py#L50),
[arguments.py:55–57](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/src/common/arguments.py#L55-L57),
[train.py:105–116](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/src/models/train.py#L105-L116),
[utils.py:104–108](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/src/models/utils.py#L104-L108)

[Старый JSON](../first_run/b55880a0.json) · [Новый JSON](b55880a0.json) ·
[Команда и SHA-256 PDF](b55880a0.run.json).

## Версия CLI

`0.1.0`, рабочее дерево поверх Git HEAD
`bae35a189b48c0b77559888f000a590a49eb1f19`. Точный снимок этого прогона
имеет SHA-256 `41b9c698aa315509136bce1252b184830fdb0abcc798fa8a15c85e998434ab26`.
[Версия среды](version.json) · [Архив исходников](cli_snapshot.tar.gz) ·
[Контрольные суммы](cli_snapshot.sha256).

Правила, эталонная разметка, исходный снимок и результаты первого прогона
сохранены. Исследовательский код не запускался.
