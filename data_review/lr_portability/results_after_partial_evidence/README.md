# GCN-Align — сохранение частичного доказательства LR

Кейс разработки `2c296ba3`. [Предыдущий прогон](../results_after_group_support/2c296ba3.json)
и [новый результат](2c296ba3.json) сохранены отдельно. Статус
`insufficient_materials` сохранился, теперь в `code_value` записано `"20"`.

В новом прогоне `--stage 'ZH–EN'` точно указывает на вариант эксперимента.
Ранее использованный текст `Experiment Settings` обозначает раздел статьи.
Конкретный вариант в README обозначен литералом `zh_en` в аргументе запуска.

- **Статья.** [ACL D18-1032, стр. 6 PDF, печатная стр. 354](https://aclanthology.org/D18-1032.pdf#page=6)
  описывает SGD и экспериментальный протокол §5.2. Численный начальный LR
  для ZH–EN в PDF не найден.
- **Команда.** [README.md:19](https://github.com/1049451037/GCN-Align/blob/4fc90c438e5a609b96df03daff170fbcf03fde94/README.md#L19)
  запускает `python train.py --lang zh_en`. Единственный переданный флаг
  выбирает язык. [train.py:17–31](https://github.com/1049451037/GCN-Align/blob/4fc90c438e5a609b96df03daff170fbcf03fde94/train.py#L17-L31)
  объявляет LR `20` и передаёт `FLAGS.lang` в `load_data`.
  [utils.py:218–226](https://github.com/1049451037/GCN-Align/blob/4fc90c438e5a609b96df03daff170fbcf03fde94/utils.py#L218-L226)
  формирует пути `data/zh_en/…` для этого аргумента.
- **Оптимизатор.** [train.py:54–56](https://github.com/1049451037/GCN-Align/blob/4fc90c438e5a609b96df03daff170fbcf03fde94/train.py#L54-L56)
  создаёт модели выравнивания. Конструктор на
  [models.py:193](https://github.com/1049451037/GCN-Align/blob/4fc90c438e5a609b96df03daff170fbcf03fde94/models.py#L193)
  передаёт `FLAGS.learning_rate` в TensorFlow `GradientDescentOptimizer`.

Значение 20 относится к опубликованному рецепту на коммите
`4fc90c438e5a609b96df03daff170fbcf03fde94`. Для сравнения нужен численный
LR из статьи.

## Воспроизведение

Из корня проекта:

```bash
work=$(mktemp -d)
commit=4fc90c438e5a609b96df03daff170fbcf03fde94
curl -fL https://aclanthology.org/D18-1032.pdf -o "$work/paper.pdf"
printf '%s  %s\n' 89a88c497c757eb5cedc7645f0e0616ffb269094e0866825f62fe36c0dafb8ed "$work/paper.pdf" | sha256sum -c -
git init --bare "$work/git"
git -C "$work/git" fetch --depth=1 --no-tags https://github.com/1049451037/GCN-Align.git "$commit"
.venv/bin/python -m ai_auditor --pdf "$work/paper.pdf" --repo "$work/git" --commit "$commit" --stage 'ZH–EN'
```

[Лог команды прогона](2c296ba3.command.log). Исследовательский код не
запускался. Все 25 тестов прошли.
