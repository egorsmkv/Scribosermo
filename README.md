# DeepSpeech Polyglot

_This project is build upon the paper [German End-to-end Speech Recognition based on DeepSpeech](https://www.researchgate.net/publication/336532830_German_End-to-end_Speech_Recognition_based_on_DeepSpeech)._
_Original paper code can be found [here](https://github.com/AASHISHAG/deepspeech-german)._

This project uses [Mozilla DeepSpeech](https://github.com/mozilla/DeepSpeech) to train the speech-to-text network.
But it also has some experimental support for [Wav2Letter](https://github.com/facebookresearch/wav2letter/).

<div align="center">
    <img src="media/deep_speech_architecture.png" alt="deepspeech graph" width="45%"/>
</div>

<br/>

[![pipeline status](https://gitlab.com/Jaco-Assistant/deepspeech-polyglot/badges/master/pipeline.svg)](https://gitlab.com/Jaco-Assistant/deepspeech-polyglot/-/commits/master)
[![coverage report](https://gitlab.com/Jaco-Assistant/deepspeech-polyglot/badges/master/coverage.svg)](https://gitlab.com/Jaco-Assistant/deepspeech-polyglot/-/commits/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![code style: prettier](https://img.shields.io/badge/code_style-prettier-ff69b4.svg)](https://github.com/prettier/prettier)
[![code complexity](https://gitlab.com/Jaco-Assistant/deepspeech-polyglot/-/jobs/artifacts/master/raw/badges/rcc.svg?job=analysis)](https://gitlab.com/Jaco-Assistant/deepspeech-polyglot/-/commits/master)

<br/>

## Datasets

#### German (de)

- [BAS-Formtask](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/FORMTASK/FORMTASK.2.php) ~18h
- [BAS-Sprecherinnen](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/SprecherInnen/SprecherInnen.1.php) ~2h
- [Mozilla Common Voice](https://voice.mozilla.org/) ~701h
- [Common Voice Single Words](https://voice.mozilla.org/) ~9h, included in the main dataset
- [CSS10](https://www.kaggle.com/bryanpark/german-single-speaker-speech-dataset) ~16h
- GoogleWavenet ~165h, artificial training data generated with the google text to speech service
- Gothic ~39h, extracted from Gothic 1-3 games
- [Kurzgesagt](https://www.youtube.com/c/KurzgesagtDE/videos) ~8h
- [LinguaLibre](https://lingualibre.org/wiki/LinguaLibre:Main_Page) ~4h
- [M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) ~234h
- MussteWissen [Deutsch](https://www.youtube.com/c/musstewissenDeutsch/videos) [Mathe](https://www.youtube.com/c/musstewissenMathe/videos) [Physik](https://www.youtube.com/c/musstewissenPhysik/videos) [Chemie](https://www.youtube.com/c/musstewissenChemie/videos) ~11h
- [PULS-Reportage](https://www.youtube.com/puls/videos) ~12h
- [Spoken Wikipedia Corpora (SWC)](https://nats.gitlab.io/swc/) ~248h
- [Tatoeba](https://tatoeba.org/deu/sentences/search?query=&from=deu&to=und&user=&orphans=no&unapproved=no&has_audio=yes&tags=&list=&native=&trans_filter=limit&trans_to=und&trans_link=&trans_user=&trans_orphan=&trans_unapproved=&trans_has_audio=&sort_reverse=&sort=relevance) ~7h
- [TerraX](https://www.youtube.com/c/terra-x/videos) ~38h
- [German Distant Speech Corpus (TUDA-De)](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/acoustic-models.html) ~185h
- [Voxforge](http://www.voxforge.org/home/forums/other-languages/german/open-speech-data-corpus-for-german) ~32h
- [Y-Kollektiv](https://www.youtube.com/c/ykollektiv/videos) ~52h
- [Zamia-Speech](https://goofy.zamia.org/zamia-speech/corpora/zamia_de/) ~19h

#### Spanish (es)

- [Mozilla Common Voice](https://voice.mozilla.org/) ~390h
- [CSS10](https://www.kaggle.com/bryanpark/spanish-single-speaker-speech-dataset) ~24h
- [LinguaLibre](https://lingualibre.org/wiki/LinguaLibre:Main_Page) ~1h
- [M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) ~108h
- [Tatoeba](https://tatoeba.org/spa/sentences/search?query=&from=spa&to=und&user=&orphans=no&unapproved=no&has_audio=yes&tags=&list=&native=&trans_filter=limit&trans_to=und&trans_link=&trans_user=&trans_orphan=&trans_unapproved=&trans_has_audio=&sort_reverse=&sort=relevance) ~59h
- [Voxforge](http://www.voxforge.org/home/) ~52h

#### French (fr)

- [Mozilla Common Voice](https://voice.mozilla.org/) ~546h
- [CSS10](https://www.kaggle.com/bryanpark/french-single-speaker-speech-dataset) ~19h
- [LinguaLibre](https://lingualibre.org/wiki/LinguaLibre:Main_Page) ~40h
- [M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) ~184h
- [Tatoeba](https://tatoeba.org/fra/sentences/search?query=&from=fra&to=und&user=&orphans=no&unapproved=no&has_audio=yes&tags=&list=&native=&trans_filter=limit&trans_to=und&trans_link=&trans_user=&trans_orphan=&trans_unapproved=&trans_has_audio=&sort_reverse=&sort=relevance) ~2h
- [Voxforge](http://www.voxforge.org/home/) ~37h

#### Italian (it)

- [Mozilla Common Voice](https://voice.mozilla.org/) ~149h
- [LinguaLibre](https://lingualibre.org/wiki/LinguaLibre:Main_Page) ~0h
- [M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) ~127h
- [Voxforge](http://www.voxforge.org/home/) ~20h

#### Polish (pl)

- [Mozilla Common Voice](https://voice.mozilla.org/) ~113h
- [LinguaLibre](https://lingualibre.org/wiki/LinguaLibre:Main_Page) ~2h
- [M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) ~54h

#### Noise

- [Freesound Dataset Kaggle 2019](https://zenodo.org/record/3612637#.Xjq7OuEo9rk) ~103h
- [RNNoise](https://people.xiph.org/~jm/demo/rnnoise/) ~44h
- [Zamia-Noise](http://goofy.zamia.org/zamia-speech/corpora/noise.tar.xz) ~5h

<br>

#### Links

- [Forschergeist](https://forschergeist.de/archiv/) ~100-150h, no aligned transcriptions
- [Verbmobil + Others](https://www.phonetik.uni-muenchen.de/Bas/BasKorporaeng.html) and [Here](https://clarin.phonetik.uni-muenchen.de/BASRepository/index.php), seems to be paid only
- [Many different languages](https://github.com/JRMeyer/open-speech-corpora)
- [Many different languages](https://www.clarin.eu/resource-families/spoken-corpora), most with login or non commercial
- GerTV1000h German Broadcast corpus and Difficult Speech Corpus (DiSCo), no links found
- [Bundestag Plenarsitzungen](https://www.bundestag.de/mediathek), very much data, but not aligned
- [LibriVox](https://librivox.org/search?primary_key=0&search_category=language&search_page=1&search_form=get_results), very much data but not aligned, get transcriptions from [Project Gutenberg](https://www.gutenberg.org/)

<br/>

## Usage

#### General infos

File structure will look as follows:

```text
my_deepspeech_folder
    checkpoints
    data_original
    data_prepared
    DeepSpeech
    deepspeech-polyglot    <- This repository
```

Clone DeepSpeech and build container:

```bash
# git clone https://github.com/mozilla/DeepSpeech.git
git clone https://github.com/DanBmh/DeepSpeech.git

# checkout older version because I have the feeling that augmentations did work better
cd DeepSpeech && git checkout before_new_augs2 && cd ..

# or for for testing with noise
cd DeepSpeech && git checkout noiseaugmaster && cd ..

cd DeepSpeech && make Dockerfile.train && cd ..
docker build -f DeepSpeech/Dockerfile.train -t mozilla_deepspeech DeepSpeech/
```

Build and run our docker container:

```bash
docker build -t deepspeech_polyglot deepspeech-polyglot/

./deepspeech-polyglot/run_container.sh
```

<br/>

#### Download and prepare voice data

Download datasets (Run in container):

```bash
export LANGUAGE="de"

# Base command
python3 /DeepSpeech/deepspeech-polyglot/preprocessing/download_data.py --language "${LANGUAGE}" --target_path "/DeepSpeech/data_original/${LANGUAGE}/" [InsertDatasetHere]

# Check above dataset chapter for tested languages
# Some flags support more languages, just test it, you will get an error message if it's not supported or existing
# With '--lingualibre' and '--tatoeba' it's often possible to add support by finding out the language conversion codes
--swc --tuda --zamia_speech
--common_voice --mailabs --voxforge
--lingualibre --tatoeba

# Download common-voice single-word datasets
python3 /DeepSpeech/deepspeech-polyglot/corpora/get_datasets.py "/DeepSpeech/data_original/xx/" --cv_singleword
mkdir /DeepSpeech/data_original/${LANGUAGE}/cv_singleword/
mv /DeepSpeech/data_original/xx/cv-corpus-5-singleword/${LANGUAGE}/ /DeepSpeech/data_original/${LANGUAGE}/cv_singleword/${LANGUAGE}/

# Downloads from youtube
# If you run into space issues you can delete the original dataset after dataset peparation
python3 /DeepSpeech/deepspeech-polyglot/corpora/get_datasets.py "/DeepSpeech/data_original/de/" [InsertDatasetHere]

# Choose one of those datasets
--kurzgesagt --musstewissen --pulsreportage --terrax --ykollektiv

```

Download css10 german/spanish/french dataset (Requires kaggle account, see links above) \
Extract and move it to datasets directory (data_original/\${LANGUAGE}/css_ten/) \
It seems the files are saved all twice, so remove the duplicate folders

<br/>

Prepare datasets, this may take some time (Run in container):

```bash
export LANGUAGE="de"

python3 /DeepSpeech/deepspeech-polyglot/corpora/prepare_datasets.py --bas_formtask /DeepSpeech/data_original/de/FORMTASK/ /DeepSpeech/data_prepared/de/bas_formtask/
python3 /DeepSpeech/deepspeech-polyglot/corpora/prepare_datasets.py --bas_sprecherinnen /DeepSpeech/data_original/de/SprecherInnen/ /DeepSpeech/data_prepared/de/bas_sprecherinnen/
python3 /DeepSpeech/deepspeech-polyglot/preprocessing/prepare_data.py --common_voice /DeepSpeech/data_original/${LANGUAGE}/common_voice/${LANGUAGE}/ /DeepSpeech/data_prepared/${LANGUAGE}/common_voice/
python3 /DeepSpeech/deepspeech-polyglot/preprocessing/prepare_data.py --css_ten /DeepSpeech/data_original/${LANGUAGE}/css_ten/ /DeepSpeech/data_prepared/${LANGUAGE}/css_ten/
python3 /DeepSpeech/deepspeech-polyglot/preprocessing/prepare_data.py --common_voice /DeepSpeech/data_original/${LANGUAGE}/cv_singleword/${LANGUAGE}/ /DeepSpeech/data_prepared/${LANGUAGE}/cv_singleword/
python3 /DeepSpeech/deepspeech-polyglot/preprocessing/prepare_data.py --lingualibre /DeepSpeech/data_original/${LANGUAGE}/lingualibre/ /DeepSpeech/data_prepared/${LANGUAGE}/lingualibre/
python3 /DeepSpeech/deepspeech-polyglot/preprocessing/prepare_data.py --mailabs /DeepSpeech/data_original/${LANGUAGE}/mailabs/ /DeepSpeech/data_prepared/${LANGUAGE}/mailabs/
python3 /DeepSpeech/deepspeech-polyglot/preprocessing/prepare_data.py --swc /DeepSpeech/data_original/${LANGUAGE}/swc/ /DeepSpeech/data_prepared/${LANGUAGE}/swc/
python3 /DeepSpeech/deepspeech-polyglot/preprocessing/prepare_data.py --tatoeba /DeepSpeech/data_original/${LANGUAGE}/tatoeba/ /DeepSpeech/data_prepared/${LANGUAGE}/tatoeba/
python3 /DeepSpeech/deepspeech-polyglot/preprocessing/prepare_data.py --tuda /DeepSpeech/data_original/${LANGUAGE}/tuda/ /DeepSpeech/data_prepared/${LANGUAGE}/tuda/
python3 /DeepSpeech/deepspeech-polyglot/preprocessing/prepare_data.py --voxforge /DeepSpeech/data_original/${LANGUAGE}/voxforge/ /DeepSpeech/data_prepared/${LANGUAGE}/voxforge/
python3 /DeepSpeech/deepspeech-polyglot/corpora/prepare_datasets.py --youtube_dir /DeepSpeech/data_original/${LANGUAGE}/kurzgesagt/ /DeepSpeech/data_prepared/${LANGUAGE}/kurzgesagt/
python3 /DeepSpeech/deepspeech-polyglot/corpora/prepare_datasets.py --youtube_dir /DeepSpeech/data_original/${LANGUAGE}/musstewissen_deutsch/ /DeepSpeech/data_prepared/${LANGUAGE}/musstewissen_deutsch/
python3 /DeepSpeech/deepspeech-polyglot/corpora/prepare_datasets.py --youtube_dir /DeepSpeech/data_original/${LANGUAGE}/musstewissen_mathe/ /DeepSpeech/data_prepared/${LANGUAGE}/musstewissen_mathe/
python3 /DeepSpeech/deepspeech-polyglot/corpora/prepare_datasets.py --youtube_dir /DeepSpeech/data_original/${LANGUAGE}/musstewissen_physik/ /DeepSpeech/data_prepared/${LANGUAGE}/musstewissen_physik/
python3 /DeepSpeech/deepspeech-polyglot/corpora/prepare_datasets.py --youtube_dir /DeepSpeech/data_original/${LANGUAGE}/musstewissen_chemie/ /DeepSpeech/data_prepared/${LANGUAGE}/musstewissen_chemie/
python3 /DeepSpeech/deepspeech-polyglot/corpora/prepare_datasets.py --youtube_dir /DeepSpeech/data_original/${LANGUAGE}/pulsreportage/ /DeepSpeech/data_prepared/${LANGUAGE}/pulsreportage/
python3 /DeepSpeech/deepspeech-polyglot/corpora/prepare_datasets.py --youtube_dir /DeepSpeech/data_original/${LANGUAGE}/terrax/ /DeepSpeech/data_prepared/${LANGUAGE}/terrax/
python3 /DeepSpeech/deepspeech-polyglot/corpora/prepare_datasets.py --youtube_dir /DeepSpeech/data_original/${LANGUAGE}/ykollektiv/ /DeepSpeech/data_prepared/${LANGUAGE}/ykollektiv/
python3 /DeepSpeech/deepspeech-polyglot/preprocessing/prepare_data.py --zamia_speech /DeepSpeech/data_original/${LANGUAGE}/zamia_speech/ /DeepSpeech/data_prepared/${LANGUAGE}/zamia_speech/

# To combine multiple datasets run:
python3 /DeepSpeech/deepspeech-polyglot/preprocessing/combine_datasets.py /DeepSpeech/data_prepared/de/ --tuda --voxforge

# To combine not only train, dev, test and all csv files run (not recommended because it's very slow):
python3 deepspeech-polyglot/pre-processing/prepare_data.py --tuda data_original/de/tuda/ --voxforge data_original/de/voxforge/ data_prepared/de/tuda_voxforge/

# Or to combine specific csv files:
python3 /DeepSpeech/deepspeech-polyglot/preprocessing/combine_datasets.py "" --files_output /DeepSpeech/data_prepared/${LANGUAGE}/cmv/train_mix.csv --files "/DeepSpeech/data_prepared/${LANGUAGE}/common_voice/train_s.csv /DeepSpeech/data_prepared/${LANGUAGE}/mailabs/all.csv /DeepSpeech/data_prepared/${LANGUAGE}/voxforge/train.csv"


# To shuffle and replace non alphabet characters and clean the files run (repeat for all 3 csv files):
python3 /DeepSpeech/deepspeech-polyglot/preprocessing/dataset_operations.py /DeepSpeech/data_prepared/${LANGUAGE}/voxforge/train.csv /DeepSpeech/data_prepared/${LANGUAGE}/voxforge/train_azce.csv --replace --shuffle --clean --exclude


# To split tuda into the correct train, dev and test splits run:
# (you will have to rename the [train/dev/test]_s.csv files before combining them with other datasets)
python3 deepspeech-polyglot/preprocessing/split_dataset.py /DeepSpeech/data_prepared/de/tuda/all.csv --tuda --file_appendix _s
# To split common-voice into the correct train, dev and test splits run:
python3 deepspeech-polyglot/preprocessing/split_dataset.py /DeepSpeech/data_prepared/${LANGUAGE}/common_voice/all.csv --common_voice --common_voice_org /DeepSpeech/data_original/${LANGUAGE}/common_voice/${LANGUAGE}/ --file_appendix _s
```

Preparation times for german datasets using Intel i7-8700K:

- voxforge: some seconds
- tuda: some minutes
- mailabs: ~20min
- common_voice: ~12h
- swc: ~6h

<br/>

#### Download and prepare noise data

You have to merge https://github.com/mozilla/DeepSpeech/pull/2622 for testing with noise, or use my noiseaugmaster branch. \
Run in container:

```bash
cd data_original/noise/

# Download freesound data:
wget https://zenodo.org/record/3612637/files/FSDKaggle2019.audio_test.zip?download=1 -O FSDKaggle2019.audio_test.zip
wget https://zenodo.org/record/3612637/files/FSDKaggle2019.audio_train_curated.zip?download=1 -O FSDKaggle2019.audio_train_curated.zip
wget https://zenodo.org/record/3612637/files/FSDKaggle2019.audio_train_noisy.z01?download=1 -O FSDKaggle2019.audio_train_noisy.z01
wget https://zenodo.org/record/3612637/files/FSDKaggle2019.audio_train_noisy.z02?download=1 -O FSDKaggle2019.audio_train_noisy.z02
wget https://zenodo.org/record/3612637/files/FSDKaggle2019.audio_train_noisy.z03?download=1 -O FSDKaggle2019.audio_train_noisy.z03
wget https://zenodo.org/record/3612637/files/FSDKaggle2019.audio_train_noisy.z04?download=1 -O FSDKaggle2019.audio_train_noisy.z04
wget https://zenodo.org/record/3612637/files/FSDKaggle2019.audio_train_noisy.z05?download=1 -O FSDKaggle2019.audio_train_noisy.z05
wget https://zenodo.org/record/3612637/files/FSDKaggle2019.audio_train_noisy.z06?download=1 -O FSDKaggle2019.audio_train_noisy.z06
wget https://zenodo.org/record/3612637/files/FSDKaggle2019.audio_train_noisy.zip?download=1 -O FSDKaggle2019.audio_train_noisy.zip

# Merge the seven parts:
zip -s 0 FSDKaggle2019.audio_train_noisy.zip --out unsplit.zip

unzip FSDKaggle2019.audio_test.zip
unzip FSDKaggle2019.audio_train_curated.zip
unzip unsplit.zip

rm *.zip
rm *.z0*

# Download rnnoise data:
wget https://media.xiph.org/rnnoise/rnnoise_contributions.tar.gz
tar -xvzf rnnoise_contributions.tar.gz
rm rnnoise_contributions.tar.gz

# Download zamia noise
wget http://goofy.zamia.org/zamia-speech/corpora/noise.tar.xz
tar -xvf noise.tar.xz
mv noise/ zamia/
rm noise.tar.xz

# Normalize all the audio files (run with python2):
python /DeepSpeech/deepspeech-polyglot/preprocessing/normalize_noise_audio.py --from_dir /DeepSpeech/data_original/noise/ --to_dir /DeepSpeech/data_prepared/noise/ --max_sec 45

# Create csv files:
python3 /DeepSpeech/deepspeech-polyglot/preprocessing/noise_to_csv.py
python3 /DeepSpeech/deepspeech-polyglot/preprocessing/split_dataset.py /DeepSpeech/data_prepared/noise/all.csv  --split "70|15|15"
```

<br/>

#### Create the language model

Download and prepare the text corpora [tuda](http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/), [europarl+news](https://www.statmt.org/wmt13/translation-task.html):

```bash
export LANGUAGE="de"
mkdir data_original/texts/ && mkdir data_original/texts/${LANGUAGE}/
mkdir data_prepared/texts/ && mkdir data_prepared/texts/${LANGUAGE}/
cd data_original/texts/${LANGUAGE}/

# German
wget "http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/German_sentences_8mil_filtered_maryfied.txt.gz" -O tuda_sentences.txt.gz
gzip -d tuda_sentences.txt.gz

# German or French or Spanish
wget "https://www.statmt.org/wmt13/training-monolingual-nc-v8.tgz" -O news-commentary.tgz
tar zxvf news-commentary.tgz && mv training/news-commentary-v8.${LANGUAGE} news-commentary-v8.${LANGUAGE}
rm news-commentary.tgz && rm -r training/
wget "https://www.statmt.org/wmt13/training-monolingual-europarl-v7.tgz" -O europarl.tgz
tar zxvf europarl.tgz && mv training/europarl-v7.${LANGUAGE} europarl-v7.${LANGUAGE}
rm europarl.tgz && rm -r training/
# If you have enough space you can also download the other years (2007-2011)
wget "https://www.statmt.org/wmt13/training-monolingual-news-2012.tgz" -O news-2012.tgz
tar zxvf news-2012.tgz && mv training-monolingual/news.2012.${LANGUAGE}.shuffled news.2012.${LANGUAGE}
rm news-2012.tgz && rm -r training-monolingual/

# Collect and normalize the sentences
python3 /DeepSpeech/deepspeech-polyglot/preprocessing/prepare_vocab.py /DeepSpeech/data_prepared/texts/${LANGUAGE}/clean_vocab.txt \
    --source_dir /DeepSpeech/data_original/texts/${LANGUAGE}/ --training_csv /DeepSpeech/data_prepared/${LANGUAGE}/voxforge/train.csv
```

Generate scorer (Run in container):

```bash
export LANGUAGE="de"

python3 /DeepSpeech/data/lm/generate_lm.py --input_txt /DeepSpeech/data_prepared/texts/${LANGUAGE}/clean_vocab.txt --output_dir /DeepSpeech/data_prepared/texts/${LANGUAGE}/ \
    --top_k 500000 --kenlm_bins /DeepSpeech/native_client/kenlm/build/bin/ --arpa_order 5 --max_arpa_memory "85%" --arpa_prune "0|0|1" --binary_a_bits 255 --binary_q_bits 8 --binary_type trie --discount_fallback

/DeepSpeech/data/lm/generate_scorer_package --alphabet /DeepSpeech/deepspeech-polyglot/data/alphabet_${LANGUAGE}.txt --lm /DeepSpeech/data_prepared/texts/${LANGUAGE}/lm.binary \
    --vocab /DeepSpeech/data_prepared/texts/${LANGUAGE}/vocab-500000.txt --package /DeepSpeech/data_prepared/texts/${LANGUAGE}/kenlm_${LANGUAGE}.scorer --default_alpha 0.8223176270809696 --default_beta 0.25566134318440037

# Optimized scorer alpha and beta values:
# English (taken from DeepSpeech repo): --default_alpha 0.931289039105002 --default_beta 1.1834137581510284
# German: --default_alpha 0.7842902115058261 --default_beta 0.6346150241906542
# Spanish: --default_alpha 0.749166959347089 --default_beta 1.6627453128820517
# French: --default_alpha 0.9000153993017823 --default_beta 2.478779501401466
# Italian: --default_alpha 0.910619981788069 --default_beta 0.15660475671195578
# Polish: --default_alpha 1.3060110864019918 --default_beta 3.5010876706821334
```

<br/>

#### Fix some issues

Following issues may or may not occur in your trainings, so try it first without changing the scripts.

<br/>

For me only training with voxforge worked at first. With tuda dataset I got an error: \
"Invalid argument: Not enough time for target transition sequence"

To fix it you have to follow this [solution](https://github.com/mozilla/DeepSpeech/issues/1629#issuecomment-427423707):

```text
# Add the parameter "ignore_longer_outputs_than_inputs=True" in DeepSpeech.py (~ line 231)

# Compute the CTC loss using TensorFlow's `ctc_loss`
total_loss = tfv1.nn.ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_seq_len, ignore_longer_outputs_than_inputs=True)
```

<br/>

This will result in another error after some training steps: \
"Invalid argument: WAV data chunk '[Some strange symbol here]"

Just ignore this in the train steps:

```text
# Add another exception (tf.errors.InvalidArgumentError) in the training loop in DeepSpeech.py (~ line 602):

try:
    [...]
    session.run([train_op, global_step, loss, non_finite_files, step_summaries_op], feed_dict=feed_dict)
except tf.errors.OutOfRangeError:
    break
except tf.errors.InvalidArgumentError as e:
    print("Ignoring error:", e)
    continue
```

<br/>

Add the parameter and the ignored exception in evaluate.py file too (~ lines 73 and 118).

<br/>

To filter the files causing infinite loss:

```text
# Below this lines (DeepSpeech.py ~ line 620):

problem_files = [f.decode('utf8') for f in problem_files[..., 0]]
log_error('The following files caused an infinite (or NaN) '
          'loss: {}'.format(','.join(problem_files)))


# Add the following to save the files to excluded_files.json and stop training

sys.path.append("/DeepSpeech/deepspeech-polyglot/training/")
from filter_invalid_files import add_files_to_excluded
add_files_to_excluded(problem_files)
sys.exit(1)
```

<br/>

#### Training

Download pretrained deepspeech checkpoints.

```bash
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.8.1/deepspeech-0.8.1-checkpoint.tar.gz -P checkpoints/
tar xvfz checkpoints/deepspeech-0.8.1-checkpoint.tar.gz -C checkpoints/
rm checkpoints/deepspeech-0.8.1-checkpoint.tar.gz
```

Adjust the parameters/scripts to your needs (Run in container):

```bash
# Run a training using the english checkpoint:
/bin/bash /DeepSpeech/deepspeech-polyglot/training/train.sh /DeepSpeech/checkpoints/voxforge/ /DeepSpeech/data_prepared/de/voxforge/train_azce.csv /DeepSpeech/data_prepared/de/voxforge/dev_azce.csv /DeepSpeech/data_prepared/de/voxforge/test_azce.csv 1 /DeepSpeech/checkpoints/deepspeech-0.8.1-checkpoint/

# Run test only
# Don't forget to use the noiseaugmaster image if testing with noise
/bin/bash /DeepSpeech/deepspeech-polyglot/training/test_noise.sh /DeepSpeech/checkpoints/voxforge/ /DeepSpeech/data_prepared/voxforge/test_azce.csv

# Optimize scorer alpha and beta values
# Requires a trained checkpoint and a scorer (used alpha and beta values are not important)
/bin/bash /DeepSpeech/deepspeech-polyglot/training/optimize_scorer.sh /DeepSpeech/checkpoints/voxforge/ /DeepSpeech/data_prepared/voxforge/dev_azce.csv
```

<br/>

Other things you can do:

```bash
# Delete old model files:
rm -rf /root/.local/share/deepspeech/summaries && rm -rf /root/.local/share/deepspeech/checkpoints

# Run training without the helpful script:
python3 DeepSpeech.py --train_files data_prepared/de/voxforge/train.csv --dev_files data_prepared/de/voxforge/dev.csv --test_files data_prepared/de/voxforge/test.csv \
--scorer data_prepared/texts/de/kenlm_de.scorer --alphabet_config_path deepspeech-polyglot/data/alphabet_de.txt --test_batch_size 48 --train_batch_size 48 --dev_batch_size 48 \
--epochs 75 --learning_rate 0.0005 --dropout_rate 0.40 --use_allow_growth --use_cudnn_rnn \
--export_dir checkpoints/de/voxforge/ --checkpoint_dir /checkpoints/de/voxforge/ --summary_dir /checkpoints/de/voxforge/

# Run test only (The use_allow_growth flag fixes "cuDNN failed to initialize" error):
python3 /DeepSpeech/DeepSpeech.py --test_files /DeepSpeech/data_prepared/de/voxforge/test_azce.csv --checkpoint_dir /DeepSpeech/checkpoints/de/voxforge/ \
--scorer data_prepared/texts/de/kenlm_de.scorer --alphabet_config_path /DeepSpeech/deepspeech-polyglot/data/alphabet_de.txt --test_batch_size 36 --use_allow_growth

# Or to run a cycled training as described in the paper, run:
python3 /DeepSpeech/deepspeech-polyglot/training/cycled_training.py /DeepSpeech/checkpoints/de/voxforge/ /DeepSpeech/data_prepared/de/ _azce --voxforge
```

<br/>

Training with wav2letter:

```bash
export LANGUAGE="de"

# Build docker container
git clone https://github.com/facebookresearch/wav2letter.git
cd wav2letter && git checkout v0.2
docker build --no-cache -f ./Dockerfile-CUDA -t wav2letter .
cd ..

# Convert csv files to lst files:
python3 deepspeech-polyglot/preprocessing/ds_to_w2l.py /DeepSpeech/data_prepared/${LANGUAGE}/voxforge/train_azce.csv /DeepSpeech/data_prepared/${LANGUAGE}/w2l_voxforge/train_azce.lst

# Generate lexicon file:
python3 deepspeech-polyglot/preprocessing/create_w2l_lexicon.py /DeepSpeech/data_prepared/${LANGUAGE}/voxforge/train_azce.csv /DeepSpeech/data_prepared/${LANGUAGE}/voxforge/dev_azce.csv /DeepSpeech/data_prepared/texts/${LANGUAGE}/lexicon.txt

Run training with single gpu (Uncomment according parts in run_container.sh to run this):
mpirun -n 1 --allow-run-as-root /root/wav2letter/build/Train continue --flagsfile /root/deepspeech-polyglot/training/train.cfg
mpirun -n 1 --allow-run-as-root /root/wav2letter/build/Decoder --flagsfile /root/deepspeech-polyglot/training/decode.cfg

Continue training:
mpirun -n 1 --allow-run-as-root /root/wav2letter/build/Train continue checkpoints/w2l/voxforge_conv_glu/
```

Training time for voxforge on 2x Nvidia 1080Ti using batch size of 48 is about 01:45min per epoch.

One epoch in tuda with batch size of 12 on single gpu needs about 1:15h. With both gpus it takes about 26min. For 10 cycled training with early stops it took about 15h.

One epoch with Tuda + Voxforge + SWC + Mailabs + CommonVoice did need about 3:30h. Training for 55 epochs took 8d 6h, testing about 1h.

Training the German CCLMSTTV model with 10x augmentation took 5d7h on 8x Nvidia-V100, training the French and Spanish CCLMTV models took about a day.

<br/>

## Results

#### Paper

Some results from the findings in the paper [German End-to-end Speech Recognition based on DeepSpeech](https://www.researchgate.net/publication/336532830_German_End-to-end_Speech_Recognition_based_on_DeepSpeech):

- Mozilla 79.7%
- Voxforge 72.1%
- Tuda-De 26.8%
- Tuda-De+Mozilla 57.3%
- Tuda-De+Voxforge 15.1%
- Tuda-De+Voxforge+Mozilla 21.5%

To test their uploaded checkpoint you have to add a file `best_dev_checkpoint` next to the checkpoint files.
Insert following content:

```text
model_checkpoint_path: "best_dev-22218"
all_model_checkpoint_paths: "best_dev-22218"
```

Switch the DeepSpeech repository back to tag `v0.5.0` and build a new docker image.
Don't forget to fix the above issue in _evaluate.py_ again.
Mount the checkpoint and data directories and run:

```bash
python3 /DeepSpeech/DeepSpeech.py --test_files /DeepSpeech/data_prepared/voxforge/test_azce.csv --checkpoint_dir /DeepSpeech/checkpoints/dsg05_models/checkpoints/ \
--alphabet_config_path /DeepSpeech/checkpoints/dsg05_models/alphabet.txt --lm_trie_path /DeepSpeech/checkpoints/dsg05_models/trie --lm_binary_path /DeepSpeech/checkpoints/dsg05_models/lm.binary --test_batch_size 48
```

| Dataset            | Additional Infos                                                                                                 | Losses           | Result                       |
| ------------------ | ---------------------------------------------------------------------------------------------------------------- | ---------------- | ---------------------------- |
| Tuda + CommonVoice | used newer CommonVoice version, there may be overlaps between test and training data because of random splitting | Test: 105.747589 | WER: 0.683802, CER: 0.386331 |
| Tuda               | correct tuda test split, there may be overlaps between test and training data because of random splitting        | Test: 402.696991 | WER: 0.785655, CER: 0.428786 |

<br/>

#### This repo

Some results with a old code version (Default dropout is 0.4, learning rate 0.0005):

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

| Datasets                                                                                                                                                                                                                                  | Additional Infos                                                                                                                                                                                                                                                                                                                                                                                                | Training epochs of best model <br><br> Total training duration | Losses <br><br> Result                                                       |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| BasFormtask + BasSprecherinnen + CommonVoice + CssTen + Gothic + LinguaLibre + Kurzgesagt + Mailabs + MussteWissen + PulsReportage + SWC + Tatoeba + TerraX + Tuda + Voxforge + YKollektiv + ZamiaSpeech + 5x CV-SingleWords <br> (D17S5) | test with Voxforge + Tuda + CommonVoice others completely for training; files 10x with different augmentations; noise overlay; fixed updated rlrop; optimized german scorer; updated dataset cleaning algorithm -> include more short files; added the CV-SingleWords dataset five times because the last checkpoint had problems detecting short speech commands -> a bit more focus on training shorter words | 24 <br><br> 7d 8h (7x V100-GPU)                                | Test: 25.082140, Validation: 23.345149 <br><br> WER: 0.161870, CER: 0.068542 |

<br/>

#### Language Models and Checkpoints

By default the checkpoints are provided under the same licence as this repository,
but some of the datasets have extra conditions which also have to be applied. \
Please check this yourself for the models you want to use. Some important ones are:

- Gothic: Non commercial
- Tatoeba: Various, depending on speakers
- Voxforge: GPL
- All extractions from Youtube videos: Non commercial

<br/>

**German:** \
(WER: 0.162, Train: ~1582h, Test: ~41h) \
Checkpoints of D17S5 training, graph model and scorer with training transcriptions: [Link](https://drive.google.com/drive/folders/1oO-N-VH_0P89fcRKWEUlVDm-_z18Kbkb?usp=sharing) \
You can also find some older checkpoints there.

**Spanish:** \
(WER: 0.188, Train: ~630h, Test: ~25h) \
Checkpoints of CCLMTV training, graph model and scorer: [Link](https://drive.google.com/drive/folders/1-3UgQBtzEf8QcH2qc8TJHkUqCBp5BBmO?usp=sharing)

**French:** \
(WER: 0.229, Train: ~780h, Test: ~25h) \
Checkpoints of CCLMTV training, graph model and scorer: [Link](https://drive.google.com/drive/folders/1Nk_1uFVwM7lj2RQf4PaQOgdAdqhiKWyV?usp=sharing)

**Italian:** \
(WER: 0.248 Train: ~257h, Test: ~21h) \
Checkpoints of CLMV training, graph model and scorer: [Link](https://drive.google.com/drive/folders/1BudQv6nUvRSas69SpD9zHN-TmjGyedaK?usp=sharing)

**Polish:** \
(WER: 0.034, Train: ~157h, Test: ~6h) \
Checkpoints of CLM training, graph model and scorer: [Link](https://drive.google.com/drive/folders/1_hia1rRmmsLRrFIHANH4254KKZhY3p1c?usp=sharing)

<br/>

Create checkpoint sharing file:

```bash
export LANGUAGE="de"
export ReadFrom="d17s5_0162"
export SaveTo="d17s5"

# Copy files
cd ~/checkpoints/${LANGUAGE}/
mkdir ${SaveTo}
cp "${ReadFrom}/flags.txt" "${SaveTo}/"
cp "${ReadFrom}/best_dev_checkpoint" "${SaveTo}/"
cp "${ReadFrom}/"best_dev-* "${SaveTo}/"

# Delete english transfer learning checkpoint
rm "${SaveTo}/"best_dev-732522.*

# Compress
GZIP=-9 tar cvzf "${SaveTo}_${LANGUAGE}.tar.gz" "${SaveTo}/"
```

<br/>

## Contribution

You can contribute to this project in multiple ways:

- Add a new language:
  - Extend `data/langdicts.json` and `tests/test_text_cleaning.py`
  - Add speech datasets
  - Find text corpora for the language model
- Help to solve the open issues

- Train new models or improve the existing \
   (Requires a gpu and a lot of time, or multiple gpus and some time)
- Experiment with the language models
- Last but not least, you can also buy me a coffee. \
   And if you are using this commercially, I suggest to think about a coffee machine instead;) \
  [![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donate_SM.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=HMN45MDHCNJNQ) (PayPal)

<br/>

## Testing

Run test (Run in container):

```bash
cd /DeepSpeech/deepspeech-polyglot/ && pytest --cov=preprocessing
```
