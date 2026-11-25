# Извлечение LR из документированной shell-команды

FedOptim (`b55880a0`) с этого этапа считается **кейсом разработки**. Его
строка в исходной разметке и результаты первого отложенного прогона
сохранены как исторические данные.

| Кейс | До | После |
| --- | --- | --- |
| FedOptim `b55880a0` | [insufficient_materials](../../held_out/diagnostic_rerun/b55880a0.json), `paper_value: null`, `code_value: null` | [insufficient_materials](b55880a0.json), `paper_value: null`, `code_value: "0.01"` |
| GCN-Align `2c296ba3` | [insufficient_materials](../results_after_partial_evidence/2c296ba3.json), `paper_value: null`, `code_value: "20"` | [Тот же JSON](2c296ba3.json) |

## FedOptim

Коммит `381a63ea4317f5aeacb5f1d7e36a7377450e9eae`. В
[PDF arXiv:2202.00280v1, стр. 7 и 22](https://arxiv.org/pdf/2202.00280v1#page=7)
описаны standalone LBGM и настройка экспериментов. Численный начальный
LR для указанной постановки в статье не найден.

1. [README.md:26,38–39](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/README.md#L25-L39)
   задаёт рабочий каталог `src` и шаблон запуска standalone LBGM.
   `--shell-recipe cnn cifar 3` фиксирует вариант архитектуры, ветку
   датасета и число классов на worker. В опубликованном сценарии
   построения графика non-IID фигурирует именно вариант CIFAR с
   `non_iid_3`. [expt_1_standalone.sh:25–35](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/src/sh/expt_1_standalone.sh#L25-L35)
2. В выбранном скрипте `$1` выбирает функцию `cifar()`, а `n=$2`
   задаёт число классов для `--non-iid $n`. Шесть активных команд
   для разных порогов LBGM задают `--optim sgd --lr 1e-2`.
   [train_lbgm_cnn.sh:25–32](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/src/sh/train_lbgm_cnn.sh#L25-L32),
   [привязка параметров:95–97](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/src/sh/train_lbgm_cnn.sh#L95-L97)
3. Парсер читает `--lr` как float, а `--non-iid` как int.
   Объект аргументов переносит эти поля. Путь к входным данным включает
   `n_classes_per_node_3`: training script загружает подготовленное
   разбиение по этому пути. [argparser.py:50](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/src/common/argparser.py#L50),
   [argparser.py:71–72](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/src/common/argparser.py#L71-L72),
   [common/utils.py:127–130](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/src/common/utils.py#L127-L130),
   [train_federated.py:44–51](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/src/train_federated.py#L44-L51)
4. `fl_train` передаёт аргументы локальному worker. Тот вызывает
   `get_optim`, где ветка SGD использует `lr=args.lr`.
   [models/train.py:300–309](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/src/models/train.py#L300-L309),
   [models/train.py:105–116](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/src/models/train.py#L105-L116),
   [models/utils.py:104–108](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/src/models/utils.py#L104-L108)

## Повторение

Из корня проекта:

```bash
work=$(mktemp -d)
commit=381a63ea4317f5aeacb5f1d7e36a7377450e9eae
curl -fL https://arxiv.org/pdf/2202.00280v1 -o "$work/paper.pdf"
printf '%s  %s\n' 24c25eeebc9b2ef49f8c35d325e42bb5d15569988a2dbc2d7ad1b724e75ec71c "$work/paper.pdf" | sha256sum -c -
git init --bare "$work/git"
git -C "$work/git" fetch --depth=1 --no-tags https://github.com/shams-sam/FedOptim.git "$commit"
.venv/bin/python -m ai_auditor --check initial-lr --pdf "$work/paper.pdf" \
  --repo "$work/git" --commit "$commit" --stage 'Standalone LBGM' --shell-recipe cnn cifar 3
```

[Команда и SHA-256 FedOptim](b55880a0.run.json) ·
[Команда и SHA-256 GCN-Align](2c296ba3.run.json) ·
[Контрольные суммы исходников CLI](cli_sources.sha256).

Все 28 тестов прошли. Shell-скрипты и исследовательский Python-код читались
как текст или AST.
