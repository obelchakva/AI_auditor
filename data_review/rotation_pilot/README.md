# Пилот 90c14330 — угол поворота тестового домена

В режиме `test-rotation` аргумент `--stage` задаёт эксперимент, а
`--dataset` и `--domain` выбирают набор данных и домен. Сравниваются углы
в градусах для тестового разбиения. Обозначение домена может совпадать
с углом в статье, но угол в коде вычисляется независимо из аргумента
вызова поворота.

[Фактический JSON](90c14330.json) и [команда](90c14330.command.log).
На стр. 8 PDF подпись к таблице 2 задаёт целевые домены 0° и 90° для
Rotated MNIST, а заголовок таблицы содержит MatchDG. На закреплённом
коммите команда из `README.rst:31` выбирает датасет и конфигурацию
генерации. В `data/data_gen_mnist.py:234–243` домен 90 входит в
ветку test, на строке 81 `int(domain)` передаётся в поворот, а
строки 83 и 90 сохраняют результат до аугментации.

Команда оценки из `README.rst:44` ведёт в тестовую ветку
`test.py:249–255`. Диспетчер `utils/helper.py:288–296` выбирает
подвыборку с seed 9. В `data/mnist_loader.py:31–45` загрузчик
читает `data/datasets/mnist/rot_mnist_resnet18/test/seed_9_domain_90_org_data.pt`
и добавляет его к изображениям тестового домена. JSON описывает путь,
вычисленный статически, и опубликованный рецепт на заданном коммите.

Режим поддерживает документированные команды Python с буквальными
аргументами, ограниченный набор ветвлений и построений путей и
распознаваемую передачу повёрнутого массива в файл. При недостаточной
связи со статьёй или тестовым загрузчиком возвращается
`insufficient_materials` с доступными фрагментами.

## Воспроизведение

Из корня AI_auditor, с установленным окружением:

```bash
work=$(mktemp -d)
commit=3eee1730ae9efaae2bd763f9835ac540b9bde02a
curl -fL https://proceedings.mlr.press/v139/mahajan21b/mahajan21b.pdf -o "$work/paper.pdf"
printf '%s  %s\n' c9d9f8a4fc7fabd12534630f3268a3dff52eae611eefbebd501f78718ca07c21 "$work/paper.pdf" | sha256sum -c -
git init --bare "$work/git"
git -C "$work/git" fetch --depth=1 --no-tags https://github.com/microsoft/robustdg.git "$commit"
.venv/bin/python -m ai_auditor --check test-rotation --pdf "$work/paper.pdf" \
  --repo "$work/git" --commit "$commit" --stage MatchDG --dataset 'Rotated MNIST' --domain 90
```
