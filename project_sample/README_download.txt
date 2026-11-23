Полная загрузка SciCoQA и аудит качества

1) Установить Python 3.10+.
2) Установить зависимости:
   pip install datasets pandas pyarrow
3) Скачать inspect_scicoqa.py и запустить:
   python inspect_scicoqa.py
4) Скрипт создаст папки data/raw, data/reports, data/quarantine, data/prepared.

Важно:
- Полный исходный split real будет сохранён до любых изменений в data/raw/scicoqa_real.parquet.
- Скрипт не заполняет пропуски и не удаляет строки из raw.
- Повторяющиеся discrepancy_id, записи без обязательных ссылок/меток помещаются в quarantine.
- Несогласие GPT/Gemini отмечается отдельным флагом для проверки экспертом.
- Категории Training и Data помечаются как кандидаты в область MVP; остальные сохраняются как out_of_scope_review.
- Несколько расхождений у одной пары статья—репозиторий не считаются дубликатами.
