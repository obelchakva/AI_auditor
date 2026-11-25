# Ожидания до запуска CLI

Выборка разработки (development set), 24.11.2026. Три разных исследования
из очереди `data_review/review_queue.tsv`, все вне пилота. ID связывают
проверки с исследованиями из SciCoQA. Ожидания относятся к новой проверке
начального LR, а исходные аннотации SciCoQA в этих строках касаются других
параметров. Сопоставляются статья и опубликованный рецепт на закреплённом
коммите. Для исторических запусков журналы не проверялись.

## 3bdc7c6f — Label-Free XAI

**Ожидание: `match` (0.001 = 1e-03).** Аргумент CLI: `MNIST autoencoder`.

- [Версия ICML 2022, PMLR 162, PDF стр. 19, приложение B.2](https://proceedings.mlr.press/v162/crabbe22a/crabbe22a.pdf#page=19).
  Абзац MNIST autoencoder описывает обучение denoising autoencoder с
  размерностью латентного пространства 4 в течение 100 эпох и Adam с LR .001.
- Коммит `3f6fd5e98ab2fe431d6c15abeb5896c3c7747c2b`.
  [README:84–99](https://github.com/JonathanCrabbe/Label-Free-XAI/blob/3f6fd5e98ab2fe431d6c15abeb5896c3c7747c2b/README.md#L84-L99)
  связывает запуск модуля `mnist` с `consistency_features` из §4.1.
  [experiments/mnist.py:714–717](https://github.com/JonathanCrabbe/Label-Free-XAI/blob/3f6fd5e98ab2fe431d6c15abeb5896c3c7747c2b/experiments/mnist.py#L714-L717)
  выбирает нужную функцию, у которой
  [стр. кода 57–66](https://github.com/JonathanCrabbe/Label-Free-XAI/blob/3f6fd5e98ab2fe431d6c15abeb5896c3c7747c2b/experiments/mnist.py#L57-L66)
  задают размерность 4 и 100 эпох, а
  [83–94](https://github.com/JonathanCrabbe/Label-Free-XAI/blob/3f6fd5e98ab2fe431d6c15abeb5896c3c7747c2b/experiments/mnist.py#L83-L94)
  создают denoising autoencoder и вызывают `fit`. В
  [src/lfxai/models/images.py:163–175](https://github.com/JonathanCrabbe/Label-Free-XAI/blob/3f6fd5e98ab2fe431d6c15abeb5896c3c7747c2b/src/lfxai/models/images.py#L163-L175)
  `fit` присваивает `self.lr = 1e-03` и передаёт его в Adam.

## b019c5d8 — BANA

**Ожидание: `potential_discrepancy` (0.0001 против 0.001).**
Аргумент CLI: `Classification network`. Сравниваем только начальный LR
весов предобученного backbone при обучении классификационной сети.

- [Версия CVPR 2021, PDF стр. 5–6, §4.1, Classification network](https://openaccess.thecvf.com/content/CVPR2021/papers/Oh_Background-Aware_Pooling_and_Noise-Aware_Loss_for_Weakly-Supervised_Semantic_Segmentation_CVPR_2021_paper.pdf#page=6).
  На стр. 6 начальные LR предобученных слоёв и классификатора заданы
  как 1e-4 и 1e-3 соответственно.
- Коммит `407e92e521dc2db90055470f7ff2ccb5aa474093`.
  [README.md:30–35](https://github.com/cvlab-yonsei/BANA/blob/407e92e521dc2db90055470f7ff2ccb5aa474093/README.md#L30-L35)
  назначает `stage1.py --config-file configs/stage1.yml` обучению классификатора.
  [configs/stage1.yml:12–20](https://github.com/cvlab-yonsei/BANA/blob/407e92e521dc2db90055470f7ff2ccb5aa474093/configs/stage1.yml#L12-L20)
  задаёт предобученные веса VGG и `SOLVER.LR: 1e-3`.
  [stage1.py:133–138](https://github.com/cvlab-yonsei/BANA/blob/407e92e521dc2db90055470f7ff2ccb5aa474093/stage1.py#L133-L138)
  загружает выбранный YAML, а [stage1.py:72–84](https://github.com/cvlab-yonsei/BANA/blob/407e92e521dc2db90055470f7ff2ccb5aa474093/stage1.py#L72-L84)
  загружает веса backbone и подаёт `lr` в SGD для `params[0]`.
  [models/ClsNet.py:31–36](https://github.com/cvlab-yonsei/BANA/blob/407e92e521dc2db90055470f7ff2ccb5aa474093/models/ClsNet.py#L31-L36)
  относит к обучаемому с нуля только классификатор, а
  [155–175](https://github.com/cvlab-yonsei/BANA/blob/407e92e521dc2db90055470f7ff2ccb5aa474093/models/ClsNet.py#L155-L175)
  помещает предобученные веса в `params[0]`, отдельно от смещений.
  Эксперту остаётся подтвердить интерпретацию опубликованного рецепта.

## 2c296ba3 — GCN-Align

**Ожидание: `insufficient_materials` (статья: значение не найдено,
код: 20 в опубликованном рецепте DBP15K ZH–EN).**
Аргумент CLI: `Experiment Settings`.

- [Версия EMNLP 2018, ACL D18-1032, PDF стр. 6 (печатная 354), §4.3 и §5.2](https://aclanthology.org/D18-1032.pdf#page=6).
  В методе указан SGD, в настройках эксперимента описаны разбиение и
  метрика. Проверка полного текста PDF не дала численного начального LR
  для этой настройки.
- Коммит `4fc90c438e5a609b96df03daff170fbcf03fde94`.
  [README.md:16–20](https://github.com/1049451037/GCN-Align/blob/4fc90c438e5a609b96df03daff170fbcf03fde94/README.md#L16-L20)
  документирует запуск `train.py --lang zh_en`.
  [train.py:16–21](https://github.com/1049451037/GCN-Align/blob/4fc90c438e5a609b96df03daff170fbcf03fde94/train.py#L16-L21)
  объявляет начальный LR 20, а [models.py:181–195](https://github.com/1049451037/GCN-Align/blob/4fc90c438e5a609b96df03daff170fbcf03fde94/models.py#L181-L195)
  передаёт флаг в TensorFlow SGD для модели выравнивания.
