# Повторный прогон после поддержки `python -m`

Исходный [отчёт](../README.md) и [исходные JSON](../results/) сохранены.
Для EQFace исходный CLI из Git был дополнительно запущен на той же паре
источников. Все четыре запуска выполнялись статически.

| ID | До | После |
| --- | --- | --- |
| `47339aed`, EQFace | [potential_discrepancy](47339aed.before.json) | [potential_discrepancy](47339aed.json), 0.1 / 0.05 |
| `3bdc7c6f`, Label-Free XAI | [insufficient_materials](../results/3bdc7c6f.json) | [match](3bdc7c6f.json), 0.001 / 0.001 |
| `b019c5d8`, BANA | [insufficient_materials](../results/b019c5d8.json) | [insufficient_materials](b019c5d8.json) |
| `2c296ba3`, GCN-Align | [insufficient_materials](../results/2c296ba3.json) | [insufficient_materials](2c296ba3.json) |

Для Label-Free XAI значение `.001` найдено на стр. 19 PDF в абзаце
«MNIST autoencoder». README на стр. кода 89 документирует запуск модуля,
а строки 95–96 перечисляют два эксперимента из §4.1. На коммите
`3f6fd5e98ab2fe431d6c15abeb5896c3c7747c2b` ветки в
`experiments/mnist.py:714–719` вызывают функции, создающие одну и ту же
модель и вызывающие её `fit` (строки 86, 94, 183, 193–195).
`src/lfxai/models/images.py:174–175` присваивает `self.lr = 1e-03`
и передаёт его в Adam.

BANA по-прежнему требует связи контекста PDF между страницами и чтения
выбранного YAML. У GCN-Align численный начальный LR в статье не найден.

## Повторение

Из корня проекта три кейса можно повторить по закреплённым версиям из
[`reproduce.sh`](../reproduce.sh). Без аргументов скрипт сохранит новый
вывод в `data_review/lr_portability/replay/`.

EQFace повторяется с [PDF arXiv v1](https://arxiv.org/pdf/2105.00634v1)
и коммитом `dfbf66d41707add18613247e5c43fd5d9564956c` из
[deepcam-cn/FaceQuality](https://github.com/deepcam-cn/FaceQuality).
SHA-256 PDF в этом прогоне
`f064a2eebfda23d552347e492e7afeae640cca87eef49a80f25d8c976dd3729c`.

```bash
work=$(mktemp -d)
curl -fL https://arxiv.org/pdf/2105.00634v1 -o "$work/paper.pdf"
git init --bare "$work/git"
git -C "$work/git" fetch --depth=1 --no-tags https://github.com/deepcam-cn/FaceQuality.git dfbf66d41707add18613247e5c43fd5d9564956c
.venv/bin/python -m ai_auditor --pdf "$work/paper.pdf" --repo "$work/git" \
    --commit dfbf66d41707add18613247e5c43fd5d9564956c --stage 'Step 1'
```

Проверка `.venv/bin/python -m unittest discover -s tests -v`: 13 тестов,
все прошли.
