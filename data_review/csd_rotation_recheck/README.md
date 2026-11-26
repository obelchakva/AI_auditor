# Перенос на DataLinux и повторный аудит CSD

Домашняя ссылка `readlink -e ~/AI_auditor`, физический рабочий каталог `pwd -P` и `git rev-parse --show-toplevel` указали на `/media/obelchakva/DataLinux/AI_auditor`. Каталог `/media/obelchakva/AI_auditor` отсутствовал. Исходная и целевая копии имели HEAD `630aba8133d7721d24fdba7533e52037909e81b5`. Целевое рабочее дерево до переноса было чистым. Импорт `ai_auditor` и запуск `.venv/bin/python -m ai_auditor --help` проверены в целевом корне.

Перенесены `ai_auditor/initial_lr.py`, `paper_scope.py`, `rotation.py`, `tests/test_initial_lr.py`, `tests/test_rotation.py`, новый `ai_auditor/fire_lr.py` и каталоги `data_review/independent_evaluation`, `pdf_evidence_recheck`, `fire_lr_recheck`. Целевые `.git` и `.venv` сохранены. После контрольной точки добавлен `ai_auditor/rotation_session.py` и обновлены `rotation.py` и `tests/test_rotation.py`.

## Контрольная точка и регрессии

- После переноса прошли **48 тестов**. Семь JSON в [migration_checkpoint](migration_checkpoint) побайтно совпали с отчётом Fire. DDPM дал `match`, `0.0002 / 0.0002`. [Журнал тестов](migration_tests.log)
- После доработки прошли **50 тестов**. Два новых теста проверяют фактический угол 105° при домене 90 и отказ при подаче в оценку исходного массива. [Журнал тестов](tests.log)
- Семь исходных команд повторены. [CSD](runs/csd.json) перешёл к `match`, `90 / 90`. Остальные шесть JSON побайтно совпали с Fire, включая DDPM и отказ CLIP. Ошибок запуска **0/7**.
- [DIVA](diva.json) дал `match`, `75 / 75`. [MatchDG](90c14330.json) дал `match`, `90 / 90`, с сохранённой цепочкой записи и чтения файлов. Оба JSON побайтно совпали с предыдущим прогоном.

## Доказательства CSD

PDF и коммит сверены с [исходным манифестом](../independent_evaluation/cases.tsv). В статье стр. 6 определяет Rotated MNIST, а стр. 7 завершает определение доменов и связывает M90 с оценкой вне обучающих доменов.

`readme.org:23–26` связывает метод статьи с командой `--dataset mnist --classifier mos`. `rotation/main.py:87–102` создаёт объект и вызывает `test(full=True)`. В `rotation/ResNet.py:32–33,100–107` тестовые изображения обрабатываются для доменов из литерального списка и накапливаются в `full_test_x`. Угол вычислен из аргумента вызова SciPy на строке 10 через вызов обёртки на строке 104. Значение не выводится из имени домена.

В `rotation/ResNet.py:390–409` полный тест выбирает `full_test_x`, нарезает батчи и через `feed_dict` передаёт их в `Session.run`. На строке 198 вход используется графом с `is_training=False`. Вывод относится к свежему опубликованному рецепту без локального кэша. Существующая ветка оценки через файл сохранена. Исследовательский код и TensorFlow не запускались.

[Команды семи исходных запусков](commands.sh) · [Команды DIVA и MatchDG](rotation_controls.sh) · [Снимок CLI](cli_snapshot.tar.gz) · [SHA-256 снимка](cli_snapshot.sha256)
