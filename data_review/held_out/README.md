# Первая партия отложенной оценки

Разметка для ручной приёмки на 24.11.2026. Проверены PDF и статические
фрагменты закреплённых Git-коммитов. Статусы в [таблице TSV](cases.tsv)
предложены по первоисточникам до запуска. Они служат эталоном для последующей
оценки CLI. Сравнение касается опубликованных рецептов и выбранных
снимков, а параметры исторических запусков остаются за пределами проверки.

| Кейс | Проверяемое утверждение и область | PDF | Код на закреплённом коммите | Предложенный статус |
| --- | --- | --- | --- | --- |
| `5a92be86`, Behavior Transformers | Начальный LR оптимизатора state prior, Franka Kitchen | [arXiv:2206.11251v2, стр. 20, таблицы 4–5](https://arxiv.org/pdf/2206.11251v2#page=20). Kitchen входит в список экспериментов, общий LR BeT равен `1e-4`. | `3d9d42352c0a6fb170369e95dde4ec3cfea951`. Команда Kitchen выбирает Hydra-конфиг. `lr: 1e-4` передаётся в minGPT и становится LR групп AdamW. [README:83–89](https://github.com/notmahi/bet/blob/3d9d42352c0a6fb170369e95dde4ec3cfea951/README.md#L83-L89), [YAML:17–34](https://github.com/notmahi/bet/blob/3d9d42352c0a6fb170369e95dde4ec3cfea951/configs/train_kitchen.yaml#L17-L34), [train.py:76–89](https://github.com/notmahi/bet/blob/3d9d42352c0a6fb170369e95dde4ec3cfea951/train.py#L76-L89), [mingpt.py:182–188](https://github.com/notmahi/bet/blob/3d9d42352c0a6fb170369e95dde4ec3cfea951/models/latent_generators/mingpt.py#L182-L188), [optimizer:223–225](https://github.com/notmahi/bet/blob/3d9d42352c0a6fb170369e95dde4ec3cfea951/models/libraries/mingpt/model.py#L223-L225) | `match`, `1e-4 = 1e-4`. Различие названий Adam и AdamW относится к отдельной проверке. |
| `183bd763`, BP Benchmark | Начальный LR модели Sig2Lab ResNet1D на наборе BCG | [Scientific Data 10:149, стр. 13–14, табл. 4 и Hyperparameter tuning](https://www.nature.com/articles/s41597-023-02020-6.pdf#page=13). Для DL указан Adam, численное значение LR для этого запуска отсутствует. | `813e1a6a938e2698d348129f2bcc01de44450df9`. Команда Sig2Lab выбирает конфиг с `data_name: bcg`, `model_type: resnet1d` и `lr: 0.001`. ResNet1D наследует создание Adam из Regressor. [README:227–235](https://github.com/inventec-ai-center/bp-benchmark/blob/813e1a6a938e2698d348129f2bcc01de44450df9/README.md#L227-L235), [YAML:4–13](https://github.com/inventec-ai-center/bp-benchmark/blob/813e1a6a938e2698d348129f2bcc01de44450df9/code/train/core/config/dl/resnet/resnet_bcg.yaml#L4-L13), [YAML:33–46](https://github.com/inventec-ai-center/bp-benchmark/blob/813e1a6a938e2698d348129f2bcc01de44450df9/code/train/core/config/dl/resnet/resnet_bcg.yaml#L33-L46), [Resnet1d:11–18](https://github.com/inventec-ai-center/bp-benchmark/blob/813e1a6a938e2698d348129f2bcc01de44450df9/code/train/core/models/resnet1d.py#L11-L18), [Adam:83–87](https://github.com/inventec-ai-center/bp-benchmark/blob/813e1a6a938e2698d348129f2bcc01de44450df9/code/train/core/models/base_pl.py#L83-L87) | `insufficient_materials`. Нужен численный LR из статьи или привязанный к результату конфиг авторского запуска. |
| `b55880a0`, FedOptim | Начальный LR локального SGD в standalone LBGM, CNN на non-IID CIFAR-10 | [arXiv:2202.00280v1, стр. 7, §4 и рис. 5](https://arxiv.org/pdf/2202.00280v1#page=7), [стр. 22, приложение C.2](https://arxiv.org/pdf/2202.00280v1#page=22). Описаны эксперимент и подбор параметров, численного локального LR для этой постановки в PDF не найдено. | `381a63ea4317f5aeacb5f1d7e36a7377450e9eae`. Ветка `cifar` standalone-скрипта задаёт `--optim sgd --lr 1e-2`, код worker передаёт `args.lr` в SGD. [README:38–39](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/README.md#L38-L39), [команды:25–32](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/src/sh/train_lbgm_cnn.sh#L25-L32), [worker:105–116](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/src/models/train.py#L105-L116), [SGD:104–108](https://github.com/shams-sam/FedOptim/blob/381a63ea4317f5aeacb5f1d7e36a7377450e9eae/src/models/utils.py#L104-L108) | `insufficient_materials`. Нужен независимый от рецепта источник численного LR для серии на рис. 5. |
| `EXT-domainbed-erm-45`, DomainBed | Угол поворота в тестовом домене 45°, Rotated MNIST, ERM с выбором модели по валидации на обучающих доменах | [arXiv:2007.01434v1, стр. 18, приложение B.2](https://arxiv.org/pdf/2007.01434v1#page=18) задаёт колонку 45° для ERM. [Стр. 24, приложение C](https://arxiv.org/pdf/2007.01434v1#page=24) определяет каждый домен через угол поворота. | `b93c22a1cfc3b2428398272c1a116c8de1f4139e`. Индекс 3 в массиве углов даёт 45°. Поворот выполняется при создании `TensorDataset`; `--test_envs 3` исключает этот домен из train и сохраняет его для оценки. Выбор модели берёт метрику `env3_in_acc` для held-out домена. [README:104–115](https://github.com/facebookresearch/DomainBed/blob/b93c22a1cfc3b2428398272c1a116c8de1f4139e/README.md#L104-L115), [datasets.py:112–117](https://github.com/facebookresearch/DomainBed/blob/b93c22a1cfc3b2428398272c1a116c8de1f4139e/domainbed/datasets.py#L112-L117), [163–183](https://github.com/facebookresearch/DomainBed/blob/b93c22a1cfc3b2428398272c1a116c8de1f4139e/domainbed/datasets.py#L163-L183), [train.py:146–168](https://github.com/facebookresearch/DomainBed/blob/b93c22a1cfc3b2428398272c1a116c8de1f4139e/domainbed/scripts/train.py#L146-L168), [model_selection.py:75–93](https://github.com/facebookresearch/DomainBed/blob/b93c22a1cfc3b2428398272c1a116c8de1f4139e/domainbed/model_selection.py#L75-L93) | `match`, `45° = 45°`. Угловой датасет формируется в памяти. Проверяемый официальный снимок сделан позже публикации. |

## Состав и пробелы

Три кейса LR взяты из локальной очереди SciCoQA. Их проверяемые утверждения
сформулированы отдельно от исходных аннотаций SciCoQA. Следующий шаг —
утвердить ручные статусы в TSV. Столбцы `review_decision` и `review_notes`
оставлены для приёмки.

Для угла найден **один кейс из независимого опубликованного исследования**
и **ноль из локальной очереди** после исключений. DomainBed использует
контролируемые повороты изображений MNIST в реальном опубликованном
эксперименте. У него отдельная статья и официальный репозиторий, хотя
набор изображений родственен пилоту и работа цитируется в нём.
`EXT-domainbed-erm-45` — внешний ID, поле SciCoQA для него пустое.

Подтверждённое расхождение на этой первой партии пока отсутствует.
Синтетические примеры можно добавить отдельно: один и тот же именованный
тестовый домен с `+20°` в спецификации и `−20°` в генераторе, либо угол
`20°`, записанный только для train при наличии отдельного test-загрузчика.
Такие управляемые примеры не входят в приведённые четыре строки.

Список исключений фиксирует исследования целиком, включая альтернативные
публикации, номера версий и другие записи той же работы. Сопоставлены
идентификатор публикации, название и репозиторий, а не только ID строки.
См. [реестр исключений](excluded_studies.tsv).

## Версия для оценки

Версия пакета `0.1.0`, базовый Git HEAD
`bae35a189b48c0b77559888f000a590a49eb1f19`. Оцениваемый снимок включает
текущие незакоммиченные изменения CLI и оба файла новых проверок.
Точная версия зафиксирована [архивом](cli_snapshot.tar.gz) с SHA-256
`7b4c9f86bdf4bbd1f0cd8f07a405cf6f1ae79853a13763fa216348b801f6e1db`.
[Контрольные суммы](cli_snapshot.sha256) охватывают исходники и архив.
Среда снимка: Python 3.12.3, Git 2.43.0, зависимости в
[runtime.lock.txt](runtime.lock.txt). PDF зафиксированы URL и SHA-256 в TSV,
репозитории — полным хешем коммита.

CLI на этих кейсах не запускался. Исходные правила и материалы проекта
сохранены.
