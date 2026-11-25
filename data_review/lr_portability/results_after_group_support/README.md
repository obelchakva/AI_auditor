# Прогон после поддержки групп начального LR

Выборка разработки. Версии PDF и коммиты совпадают с
[предыдущим прогоном](../results_after_module_support/README.md).
[Исходные ожидания](../expectations.md) сохранены.

| Кейс | До | После |
| --- | --- | --- |
| EQFace `47339aed` | `potential_discrepancy` | [potential_discrepancy](47339aed.json), 0.1 / 0.05 |
| Label-Free XAI `3bdc7c6f` | `match` | [match](3bdc7c6f.json), 0.001 / 0.001 |
| BANA `b019c5d8` | `insufficient_materials` | [potential_discrepancy](b019c5d8.json), 0.0001 / 0.001 |
| GCN-Align `2c296ba3` | `insufficient_materials` | [insufficient_materials](2c296ba3.json) |

## Доказательства для BANA

- В §4.1 на [стр. 5 PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Oh_Background-Aware_Pooling_and_Noise-Aware_Loss_for_Weakly-Supervised_Semantic_Segmentation_CVPR_2021_paper.pdf#page=5)
  начинается подраздел `Classification network`. Его незавершённое
  предложение продолжается на стр. 6. До следующего подзаголовка
  предобученным слоям и классификатору соответственно сопоставлены `1e-4`
  и `1e-3`.
- На коммите `407e92e521dc2db90055470f7ff2ccb5aa474093`
  [README.md:33](https://github.com/cvlab-yonsei/BANA/blob/407e92e521dc2db90055470f7ff2ccb5aa474093/README.md#L33)
  выбирает `--config-file configs/stage1.yml`.
  [stage1.py:127–138](https://github.com/cvlab-yonsei/BANA/blob/407e92e521dc2db90055470f7ff2ccb5aa474093/stage1.py#L127-L138)
  объявляет аргумент, загружает выбранный YAML и передаёт конфиг в `main`.
  [configs/stage1.yml:17–18](https://github.com/cvlab-yonsei/BANA/blob/407e92e521dc2db90055470f7ff2ccb5aa474093/configs/stage1.yml#L17-L18)
  задаёт `SOLVER.LR: 1e-3`.
- [stage1.py:73–79](https://github.com/cvlab-yonsei/BANA/blob/407e92e521dc2db90055470f7ff2ccb5aa474093/stage1.py#L73-L79)
  загружает веса backbone, присваивает `lr = cfg.SOLVER.LR` и передаёт
  его в SGD для `params[0]`. В
  [models/ClsNet.py:31–36](https://github.com/cvlab-yonsei/BANA/blob/407e92e521dc2db90055470f7ff2ccb5aa474093/models/ClsNet.py#L31-L36)
  только классификатор отнесён к слоям, обучаемым с нуля
  (from scratch), а [155–189](https://github.com/cvlab-yonsei/BANA/blob/407e92e521dc2db90055470f7ff2ccb5aa474093/models/ClsNet.py#L155-L189)
  помещает веса предобученных слоёв в `params[0]`, отдельно от смещений.

[JSON без селектора](b019c5d8.no_group.json) показывает, что прежняя
команда BANA остаётся допустимой и требует явно указать группу.
Сравнение относится к опубликованному рецепту и требует подтверждения эксперта.

## Воспроизведение

Из корня проекта загрузите закреплённые источники и повторите BANA с селектором:

```bash
cache=$(mktemp -d)
out=$(mktemp -d)
bash data_review/lr_portability/reproduce.sh "$cache" "$out"
.venv/bin/python -m ai_auditor --pdf "$cache/bana/paper.pdf" --repo "$cache/bana/git" \
  --commit 407e92e521dc2db90055470f7ff2ccb5aa474093 \
  --stage 'Classification network' --group 'pretrained weights'
```

Команда и версия PDF для EQFace приведены в
[предыдущем отчёте](../results_after_module_support/README.md#повторение).
Файлы `*.command.log` содержат точные команды этого прогона.
Исследовательский код не запускался. Все 17 тестов прошли командой
`.venv/bin/python -m unittest discover -s tests -v`.
