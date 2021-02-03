# DeepSpeech-Polyglot

_DeepSpeech-Polyglot is a framework to train Speech-to-Text networks in different languages._

<div align="center">
    <img src="media/deepspeech1_architecture.png" alt="deepspeech1 graph" width="45%"/>
    <img src="media/quartznet_architecture.png" alt="quartznet graph" width="50%"/>
</div>

<br/>

[![pipeline status](https://gitlab.com/Jaco-Assistant/deepspeech-polyglot/badges/master/pipeline.svg)](https://gitlab.com/Jaco-Assistant/deepspeech-polyglot/-/commits/master)
[![coverage report](https://gitlab.com/Jaco-Assistant/deepspeech-polyglot/badges/master/coverage.svg)](https://gitlab.com/Jaco-Assistant/deepspeech-polyglot/-/commits/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![code style: prettier](https://img.shields.io/badge/code_style-prettier-ff69b4.svg)](https://github.com/prettier/prettier)
[![code complexity](https://gitlab.com/Jaco-Assistant/deepspeech-polyglot/-/jobs/artifacts/master/raw/badges/rcc.svg?job=analysis)](https://gitlab.com/Jaco-Assistant/deepspeech-polyglot/-/commits/master)

<br/>

## Usage

Note: This repository is focused on training STT-networks.
You can find a short and experimental inference example [here](extras/exporting/testing_pb.py),
but please solve any problems you have with it yourself.

Requirements are:

- Computer with a modern gpu and working nvidia+docker setup
- Basic knowledge in python and deep-learning
- A lot of training data in your required language \
  (preferable >100h for fine-tuning and >1000h for new languages)

#### General infos

File structure will look as follows:

```text
my_deepspeech_folder
    checkpoints
    corcua                 <- Library for datasets
    data_original
    data_prepared
    deepspeech-polyglot    <- This repository
```

Clone [corcua](https://gitlab.com/Jaco-Assistant/corcua):

```bash
git clone https://gitlab.com/Jaco-Assistant/corcua.git
```

Build and run docker container:

```bash
docker build -f deepspeech-polyglot/Containerfile -t dspol ./deepspeech-polyglot/

./deepspeech-polyglot/run_container.sh
```

<br/>

#### Download and prepare voice data

Follow [readme](preprocessing/README.md) in `preprocessing` directory for preparing the voice data.

#### Create the language model

Follow [readme](langmodel/README.md) in `langmodel` directory for generating the language model.

#### Training

Follow [readme](dspol/README.md) in `dspol` directory for training your network. \
For easier inference follow the exporting [readme](extras/exporting/README.md) in `extras/exporting` directory.

<br/>

## Datasets and Networks

You can find more details about the currently used datasets [here](preprocessing/README.md#Datasets).

|              |      |     |     |     |     |     |       |
| ------------ | ---- | --- | --- | --- | --- | --- | ----- |
| Language     | DE   | EN  | ES  | FR  | IT  | PL  | Noise |
| Duration (h) | 1619 | 982 | 710 | 837 | 299 | 169 | 152   |
| Datasets     | 17   | 1   | 6   | 6   | 4   | 3   | 3     |

<br>

Implemented networks:
[DeepSpeech1](https://arxiv.org/pdf/1412.5567.pdf),
[DeepSpeech2](https://arxiv.org/pdf/1512.02595.pdf),
[Jasper](https://arxiv.org/pdf/1904.03288.pdf),
[Quartznet](https://arxiv.org/pdf/1910.10261.pdf)

Notes on the networks:

- Not every network is fully tested, but each could be trained with one single audio file.
- Some networks might differ from their paper implementations.

Supported networks with their trainable parameter count (using English alphabet):

|         |             |             |        |              |               |
| ------- | ----------- | ----------- | ------ | ------------ | ------------- |
| Network | DeepSpeech1 | DeepSpeech2 | Jasper | Quartznet5x5 | Quartznet15x5 |
| Params  | 48.7M       | 120M        | 323M   | 6.7M         | 18.9M         |

<br>

## Pretrained Checkpoints and Language Models

By default, the checkpoints are provided under the same licence as this repository, but a lot of
datasets have extra conditions (for example non-commercial use only) which also have to be applied.
Please check this yourself for the models you want to use.

**English**:

- Quartznet5x5 (WER: 4.5%): [Link](https://www.mediafire.com/file/tooxkchx6mmp13k/qnet5.zip/file)
- Quartznet15x5 (WER: 3.7%): [Link](https://www.mediafire.com/file/8izmtnpjlwdcfye/qnet15.zip/file)
- Scorer: [Link](https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3) (to DeepSpeech)

<br/>

## Contribution

You can contribute to this project in multiple ways:

- Help to solve the open issues
- Implement new networks or augmentation options
- Train new models or improve the existing \
  (Requires a gpu and a lot of time, or multiple gpus and some time)
- Experiment with the language models
- Add a new language:

  - Extend `data/langdicts.json` and add the alphabet files
  - Append some test to `tests/test_text_cleaning.py`
  - Add speech datasets
  - Find text corpora for the language model

<br/>

## Results

| Language | Network       | Additional Infos                                                                                                | Results                                                                                                        |
| -------- | ------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| EN       | Quartznet5x5  | Results from Nvidia-Nemo, using LS-dev-clean as test dataset                                                    | WER greedy: 0.0537                                                                                             |
| EN       | Quartznet5x5  | Converted model from Nvidia-Nemo, using LS-dev-clean as test dataset                                            | Loss: 9.7666 <br> CER greedy: 0.0268 <br> CER with lm: 0.0202 <br> WER greedy: 0.0809 <br> WER with lm: 0.0506 |
| EN       | Quartznet5x5  | Pretrained model from Nvidia-Nemo, one extra epoch on LibriSpeech to reduce the different spectrogram problem   | Loss: 7.3253 <br> CER greedy: 0.0202 <br> CER with lm: 0.0163 <br> WER greedy: 0.0654 <br> WER with lm: 0.0446 |
| EN       | Quartznet5x5  | above, using LS-dev-clean as test dataset (for better comparison with results from Nemo)                        | Loss: 6.9973 <br> CER greedy: 0.0203 <br> CER with lm: 0.0159 <br> WER greedy: 0.0648 <br> WER with lm: 0.0419 |
|          |               |                                                                                                                 |                                                                                                                |
| EN       | Quartznet15x5 | Results from Nvidia-Nemo, using LS-dev-clean as test dataset                                                    | WER greedy: 0.0379                                                                                             |
| EN       | Quartznet15x5 | Converted model from Nvidia-Nemo, using LS-dev-clean as test dataset                                            | Loss: 5.8044 <br> CER greedy: 0.0160 <br> CER with lm: 0.0130 <br> WER greedy: 0.0515 <br> WER with lm: 0.0355 |
| EN       | Quartznet15x5 | Pretrained model from Nvidia-Nemo, four extra epochs on LibriSpeech to reduce the different spectrogram problem | Loss: 5.3177 <br> CER greedy: 0.0143 <br> CER with lm: 0.0128 <br> WER greedy: 0.0467 <br> WER with lm: 0.0374 |
| EN       | Quartznet15x5 | above, using LS-dev-clean as test dataset (for better comparison with results from Nemo)                        | Loss: 5.0822 <br> CER greedy: 0.0134 <br> CER with lm: 0.0110 <br> WER greedy: 0.0439 <br> WER with lm: 0.0319 |

<br/>

## Old experiments

The following experiments were run with an old version of this repository,
using the DeepSpeech-1 network from [Mozilla-DeepSpeech](https://github.com/mozilla/DeepSpeech). \
While they are outdated, some of them might still provide helpful information for training the new networks.

Old checkpoints and scorers:

- German (D17S5 training and some older checkpoints, WER: 0.128, Train: ~1582h, Test: ~41h):
  [Link](https://drive.google.com/drive/folders/1oO-N-VH_0P89fcRKWEUlVDm-_z18Kbkb?usp=sharing)
- Spanish (CCLMTV training, WER: 0.165, Train: ~660h, Test: ~25h):
  [Link](https://drive.google.com/drive/folders/1-3UgQBtzEf8QcH2qc8TJHkUqCBp5BBmO?usp=sharing)
- French (CCLMTV training, WER: 0.195, Train: ~787h, Test: ~25h):
  [Link](https://drive.google.com/drive/folders/1Nk_1uFVwM7lj2RQf4PaQOgdAdqhiKWyV?usp=sharing)
- Italian (CLMV training, WER: 0.248 Train: ~257h, Test: ~21h):
  [Link](https://drive.google.com/drive/folders/1BudQv6nUvRSas69SpD9zHN-TmjGyedaK?usp=sharing)
- Polish (CLM training, WER: 0.034, Train: ~157h, Test: ~6h):
  [Link](https://drive.google.com/drive/folders/1_hia1rRmmsLRrFIHANH4254KKZhY3p1c?usp=sharing)

<br>

First experiments: \
(Default dropout is 0.4, learning rate 0.0005):

| Dataset                                       | Additional Infos                                                                                                                                                                          | Result                                         |
| --------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Voxforge                                      |                                                                                                                                                                                           | WER: 0.676611, CER: 0.403916, loss: 82.185226  |
| Voxforge                                      | with augmentation                                                                                                                                                                         | WER: 0.624573, CER: 0.348618, loss: 74.403786  |
| Voxforge                                      | without "äöü"                                                                                                                                                                             | WER: 0.646702, CER: 0.364471, loss: 82.567413  |
| Voxforge                                      | cleaned data, without "äöü"                                                                                                                                                               | WER: 0.634828, CER: 0.353037, loss: 81.905258  |
| Voxforge                                      | above checkpoint, tested on not cleaned data                                                                                                                                              | WER: 0.634556, CER: 0.352879, loss: 81.849220  |
| Voxforge                                      | checkpoint from english deepspeech, without "äöü"                                                                                                                                         | WER: 0.394064, CER: 0.190184, loss: 49.066357  |
| Voxforge                                      | checkpoint from english deepspeech, with augmentation, without "äöü", dropout 0.25, learning rate 0.0001                                                                                  | WER: 0.338685, CER: 0.150972, loss: 42.031754  |
| Voxforge                                      | reduce learning rate on plateau, with noise and standard augmentation, checkpoint from english deepspeech, cleaned data, without "äöü", dropout 0.25, learning rate 0.0001, batch size 48 | WER: 0.320507, CER: 0.131948, loss: 39.923031  |
| Voxforge                                      | above with learning rate 0.00001                                                                                                                                                          | WER: 0.350903, CER: 0.147837, loss: 43.451263  |
| Voxforge                                      | above with learning rate 0.001                                                                                                                                                            | WER: 0.518670, CER: 0.252510, loss: 62.927200  |
| Tuda + Voxforge                               | without "äöü", checkpoint from english deepspeech, cleaned train and dev data                                                                                                             | WER: 0.740130, CER: 0.462036, loss: 156.115921 |
| Tuda + Voxforge                               | first Tuda then Voxforge, without "äöü", cleaned train and dev data, dropout 0.25, learning rate 0.0001                                                                                   | WER: 0.653841, CER: 0.384577, loss: 159.509476 |
| Tuda + Voxforge + SWC + Mailabs + CommonVoice | checkpoint from english deepspeech, with augmentation, without "äöü", cleaned data, dropout 0.25, learning rate 0.0001                                                                    | WER: 0.306061, CER: 0.151266, loss: 33.218510  |

<br/>

Some results with some older code version: \
(Default values: batch size 12, dropout 0.25, learning rate 0.0001, without "äöü", cleaned data , checkpoint from english deepspeech, early stopping, reduce learning rate on plateau, evaluation with scorer and top-500k words)

| Dataset                                       | Additional Infos                                                                                                                 | Losses                                 | Training epochs of best model | Result                       |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ----------------------------- | ---------------------------- |
| Tuda + Voxforge + SWC + Mailabs + CommonVoice | test only with Tuda + CommonVoice others completely for training, language model with training transcriptions, with augmentation | Test: 29.363405, Validation: 23.509546 | 55                            | WER: 0.190189, CER: 0.091737 |
| Tuda + Voxforge + SWC + Mailabs + CommonVoice | above checkpoint tested with 3-gram language model                                                                               | Test: 29.363405                        |                               | WER: 0.199709, CER: 0.095318 |
| Tuda + Voxforge + SWC + Mailabs + CommonVoice | above checkpoint tested on Tuda only                                                                                             | Test: 87.074394                        |                               | WER: 0.378379, CER: 0.167380 |

<br/>

Some results with some older code version: \
(Default values: batch size 36, dropout 0.25, learning rate 0.0001, without "äöü", cleaned data , checkpoint from english deepspeech, early stopping, reduce learning rate on plateau, evaluation with scorer and top-500k words, data augmentation)

| Dataset                     | Additional Infos                                                                                                                                                                                                               | Losses                                   | Training epochs of best model | Result                       |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------- | ----------------------------- | ---------------------------- |
| Voxforge                    | training from scratch                                                                                                                                                                                                          | Test: 79.124008, Validation: 81.982976   | 29                            | WER: 0.603879, CER: 0.298139 |
| Voxforge                    |                                                                                                                                                                                                                                | Test: 44.312195, Validation: 47.915317   | 21                            | WER: 0.343973, CER: 0.140119 |
| Voxforge                    | without reduce learning rate on plateau                                                                                                                                                                                        | Test: 46.160049, Validation: 48.926518   | 13                            | WER: 0.367125, CER: 0.163931 |
| Voxforge                    | dropped last layer                                                                                                                                                                                                             | Test: 49.844028, Validation: 52.722362   | 21                            | WER: 0.389327, CER: 0.170563 |
| Voxforge                    | 5 cycled training                                                                                                                                                                                                              | Test: 42.973358                          |                               | WER: 0.353841, CER: 0.158554 |
|                             |
| Tuda                        | training from scratch, correct train/dev/test splitting                                                                                                                                                                        | Test: 149.653427, Validation: 137.645307 | 9                             | WER: 0.606629, CER: 0.296630 |
| Tuda                        | correct train/dev/test splitting                                                                                                                                                                                               | Test: 103.179092, Validation: 132.243965 | 3                             | WER: 0.436074, CER: 0.208135 |
| Tuda                        | dropped last layer, correct train/dev/test splitting                                                                                                                                                                           | Test: 107.047821, Validation: 101.219325 | 6                             | WER: 0.431361, CER: 0.195361 |
| Tuda                        | dropped last two layers, correct train/dev/test splitting                                                                                                                                                                      | Test: 110.523621, Validation: 103.844562 | 5                             | WER: 0.442421, CER: 0.204504 |
| Tuda                        | checkpoint from Voxforge with WER 0.344, correct train/dev/test splitting                                                                                                                                                      | Test: 100.846367, Validation: 95.410456  | 3                             | WER: 0.416950, CER: 0.198177 |
| Tuda                        | 10 cycled training, checkpoint from Voxforge with WER 0.344, correct train/dev/test splitting                                                                                                                                  | Test: 98.007607                          |                               | WER: 0.410520, CER: 0.194091 |
| Tuda                        | random dataset splitting, checkpoint from Voxforge with WER 0.344 <br> Important Note: These results are not meaningful, because same transcriptions can occur in train and test set, only recorded with different microphones | Test: 23.322618, Validation: 23.094230   | 27                            | WER: 0.090285, CER: 0.036212 |
|                             |
| CommonVoice                 | checkpoint from Tuda with WER 0.417                                                                                                                                                                                            | Test: 24.688297, Validation: 17.460029   | 35                            | WER: 0.217124, CER: 0.085427 |
| CommonVoice                 | above tested with reduced testset where transcripts occurring in trainset were removed,                                                                                                                                        | Test: 33.376812                          |                               | WER: 0.211668, CER: 0.079157 |
| CommonVoice + GoogleWavenet | above tested with GoogleWavenet                                                                                                                                                                                                | Test: 17.653290                          |                               | WER: 0.035807, CER: 0.007342 |
| CommonVoice                 | checkpoint from Voxforge with WER 0.344                                                                                                                                                                                        | Test: 23.460932, Validation: 16.641201   | 35                            | WER: 0.215584, CER: 0.084932 |
| CommonVoice                 | dropped last layer                                                                                                                                                                                                             | Test: 24.480028, Validation: 17.505738   | 36                            | WER: 0.220435, CER: 0.086921 |
|                             |
| Tuda + GoogleWavenet        | added GoogleWavenet to train data, dev/test from Tuda, checkpoint from Voxforge with WER 0.344                                                                                                                                 | Test: 95.555939, Validation: 90.392490   | 3                             | WER: 0.390291, CER: 0.178549 |
| Tuda + GoogleWavenet        | GoogleWavenet as train data, dev/test from Tuda                                                                                                                                                                                | Test: 346.486420, Validation: 326.615474 | 0                             | WER: 0.865683, CER: 0.517528 |
| Tuda + GoogleWavenet        | GoogleWavenet as train/dev data, test from Tuda                                                                                                                                                                                | Test: 477.049591, Validation: 3.320163   | 23                            | WER: 0.923973, CER: 0.601015 |
| Tuda + GoogleWavenet        | above checkpoint tested with GoogleWavenet                                                                                                                                                                                     | Test: 3.406022                           |                               | WER: 0.012919, CER: 0.001724 |
| Tuda + GoogleWavenet        | checkpoint from english deepspeech tested with Tuda                                                                                                                                                                            | Test: 402.102661                         |                               | WER: 0.985554, CER: 0.752787 |
| Voxforge + GoogleWavenet    | added all of GoogleWavenet to train data, dev/test from Voxforge                                                                                                                                                               | Test: 45.643063, Validation: 49.620488   | 28                            | WER: 0.349552, CER: 0.143108 |
| CommonVoice + GoogleWavenet | added all of GoogleWavenet to train data, dev/test from CommonVoice                                                                                                                                                            | Test: 25.029057, Validation: 17.511973   | 35                            | WER: 0.214689, CER: 0.084206 |
| CommonVoice + GoogleWavenet | above tested with reduced testset                                                                                                                                                                                              | Test: 34.191067                          |                               | WER: 0.213164, CER: 0.079121 |

<br/>

Updated to DeepSpeech v0.7.3 and new english checkpoint: \
(Default values: See flags.txt in releases, scorer with kaldi-tuda sentences only)
(Testing with noise and speech overlay is done with older _noiseaugmaster_ branch, which implemented this functionality)

| Dataset                                               | Additional Infos                                                                                      | Losses                                 | Training epochs of best model | Result                       |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------- | ----------------------------- | ---------------------------- |
| Voxforge                                              |                                                                                                       | Test: 32.844025, Validation: 36.912005 | 14                            | WER: 0.240091, CER: 0.087971 |
| Voxforge                                              | without _freq_and_time_masking_ augmentation                                                          | Test: 33.698494, Validation: 38.071722 | 10                            | WER: 0.244600, CER: 0.094577 |
| Voxforge                                              | using new audio augmentation options                                                                  | Test: 29.280865, Validation: 33.294815 | 21                            | WER: 0.220538, CER: 0.079463 |
|                                                       |
| Voxforge                                              | updated augmentations again                                                                           | Test: 28.846869, Validation: 32.680268 | 16                            | WER: 0.225360, CER: 0.083504 |
| Voxforge                                              | test above with older _noiseaugmaster_ branch                                                         | Test: 28.831675                        |                               | WER: 0.238961, CER: 0.081555 |
| Voxforge                                              | test with speech overlay                                                                              | Test: 89.661995                        |                               | WER: 0.570903, CER: 0.301745 |
| Voxforge                                              | test with noise overlay                                                                               | Test: 53.461609                        |                               | WER: 0.438126, CER: 0.213890 |
| Voxforge                                              | test with speech and noise overlay                                                                    | Test: 79.736122                        |                               | WER: 0.581259, CER: 0.310365 |
| Voxforge                                              | second test with speech and noise to check random influence                                           | Test: 81.241333                        |                               | WER: 0.595410, CER: 0.319077 |
|                                                       |
| Voxforge                                              | add speech overlay augmentation                                                                       | Test: 28.843914, Validation: 32.341234 | 27                            | WER: 0.222024, CER: 0.083036 |
| Voxforge                                              | change snr=50:20~9m to snr=30:15~9                                                                    | Test: 28.502413, Validation: 32.236247 | 28                            | WER: 0.226005, CER: 0.085475 |
| Voxforge                                              | test above with older _noiseaugmaster_ branch                                                         | Test: 28.488537                        |                               | WER: 0.239530, CER: 0.083855 |
| Voxforge                                              | test with speech overlay                                                                              | Test: 47.783081                        |                               | WER: 0.383612, CER: 0.175735 |
| Voxforge                                              | test with noise overlay                                                                               | Test: 51.682060                        |                               | WER: 0.428566, CER: 0.209789 |
| Voxforge                                              | test with speech and noise overlay                                                                    | Test: 60.275940                        |                               | WER: 0.487709, CER: 0.255167 |
|                                                       |
| Voxforge                                              | add noise overlay augmentation                                                                        | Test: 27.940659, Validation: 31.988175 | 28                            | WER: 0.219143, CER: 0.076050 |
| Voxforge                                              | change snr=50:20~6 to snr=24:12~6                                                                     | Test: 26.588453, Validation: 31.151855 | 34                            | WER: 0.206141, CER: 0.072018 |
| Voxforge                                              | change to snr=18:9~6                                                                                  | Test: 26.311581, Validation: 30.531299 | 30                            | WER: 0.211865, CER: 0.074281 |
| Voxforge                                              | test above with older _noiseaugmaster_ branch                                                         | Test: 26.300938                        |                               | WER: 0.227466, CER: 0.073827 |
| Voxforge                                              | test with speech overlay                                                                              | Test: 76.401451                        |                               | WER: 0.499962, CER: 0.254203 |
| Voxforge                                              | test with noise overlay                                                                               | Test: 44.011471                        |                               | WER: 0.376783, CER: 0.165329 |
| Voxforge                                              | test with speech and noise overlay                                                                    | Test: 65.408264                        |                               | WER: 0.496168, CER: 0.246516 |
|                                                       |
| Voxforge                                              | speech and noise overlay                                                                              | Test: 27.101889, Validation: 31.407527 | 44                            | WER: 0.220243, CER: 0.082179 |
| Voxforge                                              | test above with older _noiseaugmaster_ branch                                                         | Test: 27.087360                        |                               | WER: 0.232094, CER: 0.080319 |
| Voxforge                                              | test with speech overlay                                                                              | Test: 46.012951                        |                               | WER: 0.362291, CER: 0.164134 |
| Voxforge                                              | test with noise overlay                                                                               | Test: 44.035809                        |                               | WER: 0.377276, CER: 0.171528 |
| Voxforge                                              | test with speech and noise overlay                                                                    | Test: 53.832214                        |                               | WER: 0.441768, CER: 0.218798 |
|                                                       |
| Tuda + Voxforge + SWC + Mailabs + CommonVoice         | test with Voxforge + Tuda + CommonVoice others completely for training, with noise and speech overlay | Test: 22.055849, Validation: 17.613633 | 46                            | WER: 0.208809, CER: 0.087215 |
| Tuda + Voxforge + SWC + Mailabs + CommonVoice         | above tested on Voxforge devdata                                                                      | Test: 16.395443                        |                               | WER: 0.163827, CER: 0.056596 |
| Tuda + Voxforge + SWC + Mailabs + CommonVoice         | optimized scorer alpha and beta on Voxforge devdata                                                   | Test: 16.395443                        |                               | WER: 0.162842                |
| Tuda + Voxforge + SWC + Mailabs + CommonVoice         | test with Voxforge + Tuda + CommonVoice, optimized scorer alpha and beta                              | Test: 22.055849                        |                               | WER: 0.206960, CER: 0.086306 |
| Tuda + Voxforge + SWC + Mailabs + CommonVoice         | scorer (kaldi-tuda) with train transcriptions, optimized scorer alpha and beta                        | Test: 22.055849                        |                               | WER: 0.134221, CER: 0.064267 |
| Tuda + Voxforge + SWC + Mailabs + CommonVoice         | scorer only out of train transcriptions, optimized scorer alpha and beta                              | Test: 22.055849                        |                               | WER: 0.142880, CER: 0.064958 |
| Tuda + Voxforge + SWC + Mailabs + CommonVoice         | scorer (kaldi-tuda + europarl + news) with train transcriptions, optimized scorer alpha and beta      | Test: 22.055849                        |                               | WER: 0.135759, CER: 0.064773 |
| Tuda + Voxforge + SWC + Mailabs + CommonVoice         | above scorer with 1m instead of 500k top words, optimized scorer alpha and beta                       | Test: 22.055849                        |                               | WER: 0.136650, CER: 0.066470 |
| Tuda + Voxforge + SWC + Mailabs + CommonVoice         | test with Tuda only                                                                                   | Test: 54.977085                        |                               | WER: 0.250665, CER: 0.103428 |
|                                                       |
| Voxforge FR                                           | speech and noise overlay                                                                              | Test: 5.341695, Validation: 12.736551  | 49                            | WER: 0.175954, CER: 0.045416 |
| CommonVoice + Css10 + Mailabs + Tatoeba + Voxforge FR | test with Voxforge + CommonVoice others completely for training, with speech and noise overlay        | Test: 20.404339, Validation: 21.920289 | 62                            | WER: 0.302113, CER: 0.121300 |
| CommonVoice + Css10 + Mailabs + Tatoeba + Voxforge ES | test with Voxforge + CommonVoice others completely for training, with speech and noise overlay        | Test: 14.521997, Validation: 22.408368 | 51                            | WER: 0.154061, CER: 0.055357 |

<br/>

Using new CommonVoice v5 releases: \
(Default values: See flags.txt in released checkpoints; using correct instead of random splits of CommonVoice; Old german scorer alpha+beta for all)

| Language | Dataset                                                                                      | Additional Infos                                                                                                                                                    | Losses                                 | Training epochs of best model | Result                       |
| -------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ----------------------------- | ---------------------------- |
| DE       | CommonVoice + CssTen + LinguaLibre + Mailabs + SWC + Tatoeba + Tuda + Voxforge + ZamiaSpeech | test with CommonVoice + Tuda + Voxforge, others completely for training; with speech and noise overlay; top-488538 scorer (words occurring at least five times)     | Test: 29.286192, Validation: 26.864552 | 30                            | WER: 0.182088, CER: 0.081321 |
| DE       | CommonVoice + CssTen + LinguaLibre + Mailabs + SWC + Tatoeba + Tuda + Voxforge + ZamiaSpeech | like above, but using each file 10x with different augmentations                                                                                                    | Test: 25.694464, Validation: 23.128045 | 16                            | WER: 0.166629, CER: 0.071999 |
| DE       | CommonVoice + CssTen + LinguaLibre + Mailabs + SWC + Tatoeba + Tuda + Voxforge + ZamiaSpeech | above checkpoint, tested on Tuda only                                                                                                                               | Test: 57.932476                        |                               | WER: 0.260319, CER: 0.109301 |
| DE       | CommonVoice + CssTen + LinguaLibre + Mailabs + SWC + Tatoeba + Tuda + Voxforge + ZamiaSpeech | optimized scorer alpha+beta                                                                                                                                         | Test: 25.694464                        |                               | WER: 0.166330, CER: 0.070268 |
| ES       | CommonVoice + CssTen + LinguaLibre + Mailabs + Tatoeba + Voxforge                            | test with CommonVoice, others completely for training; with speech and noise overlay; top-303450 scorer (words occurring at least twice)                            | Test: 25.443010, Validation: 22.686161 | 42                            | WER: 0.193316, CER: 0.093000 |
| ES       | CommonVoice + CssTen + LinguaLibre + Mailabs + Tatoeba + Voxforge                            | optimized scorer alpha+beta                                                                                                                                         | Test: 25.443010                        |                               | WER: 0.187535, CER: 0.083490 |
| FR       | CommonVoice + CssTen + LinguaLibre + Mailabs + Tatoeba + Voxforge                            | test with CommonVoice, others completely for training; with speech and noise overlay; top-316458 scorer (words occurring at least twice)                            | Test: 29.761099, Validation: 24.691544 | 52                            | WER: 0.231981, CER: 0.116503 |
| FR       | CommonVoice + CssTen + LinguaLibre + Mailabs + Tatoeba + Voxforge                            | optimized scorer alpha+beta                                                                                                                                         | Test: 29.761099                        |                               | WER: 0.228851, CER: 0.109247 |
| IT       | CommonVoice + LinguaLibre + Mailabs + Voxforge                                               | test with CommonVoice, others completely for training; with speech and noise overlay; top-51216 scorer out of train transcriptions (words occurring at least twice) | Test: 25.536196, Validation: 23.048596 | 46                            | WER: 0.249197, CER: 0.093717 |
| IT       | CommonVoice + LinguaLibre + Mailabs + Voxforge                                               | optimized scorer alpha+beta                                                                                                                                         | Test: 25.536196                        |                               | WER: 0.247785, CER: 0.096247 |
| PL       | CommonVoice + LinguaLibre + Mailabs                                                          | test with CommonVoice, others completely for training; with speech and noise overlay; top-39175 scorer out of train transcriptions (words occurring at least twice) | Test: 14.902746, Validation: 15.508280 | 53                            | WER: 0.040128, CER: 0.022947 |
| PL       | CommonVoice + LinguaLibre + Mailabs                                                          | optimized scorer alpha+beta                                                                                                                                         | Test: 14.902746                        |                               | WER: 0.034115, CER: 0.020230 |

<br/>

| Dataset                        | Additional Infos                                                                                      | Losses                                 | Training epochs of best model | Total training duration | WER       |
| ------------------------------ | ----------------------------------------------------------------------------------------------------- | -------------------------------------- | ----------------------------- | ----------------------- | --------- |
| Voxforge                       | updated rlrop; frozen transfer-learning; no augmentation; es_min_delta=0.9                            | Test: 37.707958, Validation: 41.832220 | 12 + 3                        | 42 min                  |           |
| Voxforge                       | like above; without frozen transfer-learning;                                                         | Test: 36.630890, Validation: 41.208125 | 7                             | 28 min                  |           |
| Voxforge                       | dropped last layer                                                                                    | Test: 42.516270, Validation: 47.105518 | 8                             | 28 min                  |           |
| Voxforge                       | dropped last layer; with frozen transfer-learning in two steps                                        | Test: 36.600590, Validation: 40.640134 | 14 + 8                        | 42 min                  |           |
| Voxforge                       | updated rlrop; with augmentation; es_min_delta=0.9                                                    | Test: 35.540062, Validation: 39.974685 | 6                             | 46 min                  |           |
| Voxforge                       | updated rlrop; with old augmentations; es_min_delta=0.1                                               | Test: 30.655203, Validation: 33.655750 | 9                             | 48 min                  |           |
| TerraX + Voxforge + YKollektiv | Voxforge only for dev+test but not in train; rest like above                                          | Test: 32.936977, Validation: 36.828410 | 19                            | 4:53 h                  |           |
| Voxforge                       | layer normalization; updated rlrop; with old augmentations; es_min_delta=0.1                          | Test: 57.330410, Validation: 61.025009 | 45                            | 2:37 h                  |           |
| Voxforge                       | dropout=0.3; updated rlrop; with old augmentations; es_min_delta=0.1                                  | Test: 30.353968, Validation: 33.144178 | 20                            | 1:15 h                  |           |
| Voxforge                       | es_min_delta=0.05; updated rlrop; with old augmentations                                              | Test: 29.884317, Validation: 32.944382 | 12                            | 54 min                  |           |
| Voxforge                       | fixed updated rlrop; es_min_delta=0.05; with old augmentations                                        | Test: 28.903509, Validation: 32.322064 | 34                            | 1:40 h                  |           |
| Voxforge                       | from scratch; no augmentations; fixed updated rlrop; es_min_delta=0.05                                | Test: 74.347054, Validation: 79.838900 | 28                            | 1:26 h                  | 0.38      |
| Voxforge                       | wav2letter; stopped by hand after one/two overnight runs; from scratch; no augmentations; single gpu; |                                        | 18/37                         | 16/33 h                 | 0.61/0.61 |

<br/>

| Language | Datasets                                                                                                                                                                                                                                  | Additional Infos                                                                                                                                                                                                                                                                                                                                                                                                | Training epochs of best model <br><br> Total training duration | Losses <br><br> Result                                                       |
| -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| DE       | BasFormtask + BasSprecherinnen + CommonVoice + CssTen + Gothic + LinguaLibre + Kurzgesagt + Mailabs + MussteWissen + PulsReportage + SWC + Tatoeba + TerraX + Tuda + Voxforge + YKollektiv + ZamiaSpeech + 5x CV-SingleWords <br> (D17S5) | test with Voxforge + Tuda + CommonVoice others completely for training; files 10x with different augmentations; noise overlay; fixed updated rlrop; optimized german scorer; updated dataset cleaning algorithm -> include more short files; added the CV-SingleWords dataset five times because the last checkpoint had problems detecting short speech commands -> a bit more focus on training shorter words | 24 <br><br> 7d8h <br> (7x V100-GPU)                            | Test: 25.082140, Validation: 23.345149 <br><br> WER: 0.161870, CER: 0.068542 |
| DE       | D17S5                                                                                                                                                                                                                                     | test above on CommonVoice only                                                                                                                                                                                                                                                                                                                                                                                  |                                                                | Test: 18.922359 <br><br> WER: 0.127766, CER: 0.056331                        |
| DE       | D17S5                                                                                                                                                                                                                                     | test above on Tuda only, using all test files and full dataset cleaning                                                                                                                                                                                                                                                                                                                                         |                                                                | Test: 54.675545 <br><br> WER: 0.245862, CER: 0.101032                        |
| DE       | D17S5                                                                                                                                                                                                                                     | test above on Tuda only, using official split (excluding Realtek recordings), only text replacements                                                                                                                                                                                                                                                                                                            |                                                                | Test: 39.755287 <br><br> WER: 0.186023, CER: 0.064182                        |
| FR       | CommonVoice + CssTen + LinguaLibre + Mailabs + Tatoeba + Voxforge                                                                                                                                                                         | test with CommonVoice, others completely for training; two step frozen transfer learning; augmentations only in second step; files 10x with different augmentations; noise overlay; fixed updated rlrop; optimized scorer; updated dataset cleaning algorithm -> include more short files                                                                                                                       | 14 + 34 <br><br> 5h + 5d13h <br> (7x V100-GPU)                 | Test: 24.771702, Validation: 20.062641 <br><br> WER: 0.194813, CER: 0.092049 |
| ES       | CommonVoice + CssTen + LinguaLibre + Mailabs + Tatoeba + Voxforge                                                                                                                                                                         | like above                                                                                                                                                                                                                                                                                                                                                                                                      | 15 + 27 <br><br> 5h + 3d1h <br> (7x V100-GPU)                  | Test: 21.235971, Validation: 18.722595 <br><br> WER: 0.165126, CER: 0.075567 |
