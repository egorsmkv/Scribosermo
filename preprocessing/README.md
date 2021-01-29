# Preparing voice data

Downloading and preprocessing the datasets. \
(Only some parts were already updated to the new dataset structure)

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

#### English (en)

- [LibriSpeech](http://www.openslr.org/11) ~982h

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

## Setup

Download datasets (Run in container):

```bash
export LANGUAGE="de"

# Base command
python3 /deepspeech-polyglot/preprocessing/download_data.py --language "${LANGUAGE}" --target_path "/data_original/${LANGUAGE}/" [InsertDatasetHere]

# Check above dataset chapter for tested languages
# Some flags support more languages, just test it, you will get an error message if it's not supported or existing
# With '--lingualibre' and '--tatoeba' it's often possible to add support by finding out the language conversion codes
--swc --tuda --zamia_speech
--common_voice --mailabs --voxforge
--lingualibre --tatoeba

# Download common-voice single-word datasets to "/data_original/xx/" (only possible by hand, contains multiple languages)
mkdir /data_original/${LANGUAGE}/cv_singleword/
mv /data_original/xx/cv-corpus-5-singleword/${LANGUAGE}/ /data_original/${LANGUAGE}/cv_singleword/${LANGUAGE}/

# Downloads from youtube
# If you run into space issues you can delete the original dataset after dataset peparation
python3 /deepspeech-polyglot/corpora/get_datasets.py "/data_original/de/" [InsertDatasetHere]

# Choose one of those datasets
--kurzgesagt --musstewissen --pulsreportage --terrax --ykollektiv

```

Download css10 German/Spanish/French dataset (Requires kaggle account, see links above) \
Extract and move it to datasets directory `/data_original/${LANGUAGE}/css_ten/` \
It seems the files are saved all twice, so remove the duplicate folders

<br/>

Prepare datasets, this may take some time (Run in container):

```bash
export LANGUAGE="de"

python3 /deepspeech-polyglot/corpora/prepare_datasets.py --bas_formtask /data_original/de/FORMTASK/ /data_prepared/de/bas_formtask/
python3 /deepspeech-polyglot/corpora/prepare_datasets.py --bas_sprecherinnen /data_original/de/SprecherInnen/ /data_prepared/de/bas_sprecherinnen/
python3 /deepspeech-polyglot/preprocessing/prepare_data.py --common_voice /data_original/${LANGUAGE}/common_voice/${LANGUAGE}/ /data_prepared/${LANGUAGE}/common_voice/
python3 /deepspeech-polyglot/preprocessing/prepare_data.py --css_ten /data_original/${LANGUAGE}/css_ten/ /data_prepared/${LANGUAGE}/css_ten/
python3 /deepspeech-polyglot/preprocessing/prepare_data.py --common_voice /data_original/${LANGUAGE}/cv_singleword/${LANGUAGE}/ /data_prepared/${LANGUAGE}/cv_singleword/
python3 /deepspeech-polyglot/preprocessing/prepare_data.py --lingualibre /data_original/${LANGUAGE}/lingualibre/ /data_prepared/${LANGUAGE}/lingualibre/
python3 /deepspeech-polyglot/preprocessing/prepare_data.py --mailabs /data_original/${LANGUAGE}/mailabs/ /data_prepared/${LANGUAGE}/mailabs/
python3 /deepspeech-polyglot/preprocessing/prepare_data.py --swc /data_original/${LANGUAGE}/swc/ /data_prepared/${LANGUAGE}/swc/
python3 /deepspeech-polyglot/preprocessing/prepare_data.py --tatoeba /data_original/${LANGUAGE}/tatoeba/ /data_prepared/${LANGUAGE}/tatoeba/
python3 /deepspeech-polyglot/preprocessing/prepare_data.py --tuda /data_original/${LANGUAGE}/tuda/ /data_prepared/${LANGUAGE}/tuda/
python3 /deepspeech-polyglot/preprocessing/prepare_data.py --voxforge /data_original/${LANGUAGE}/voxforge/ /data_prepared/${LANGUAGE}/voxforge/
python3 /deepspeech-polyglot/corpora/prepare_datasets.py --youtube_dir /data_original/${LANGUAGE}/kurzgesagt/ /data_prepared/${LANGUAGE}/kurzgesagt/
python3 /deepspeech-polyglot/corpora/prepare_datasets.py --youtube_dir /data_original/${LANGUAGE}/musstewissen_deutsch/ /data_prepared/${LANGUAGE}/musstewissen_deutsch/
python3 /deepspeech-polyglot/corpora/prepare_datasets.py --youtube_dir /data_original/${LANGUAGE}/musstewissen_mathe/ /data_prepared/${LANGUAGE}/musstewissen_mathe/
python3 /deepspeech-polyglot/corpora/prepare_datasets.py --youtube_dir /data_original/${LANGUAGE}/musstewissen_physik/ /data_prepared/${LANGUAGE}/musstewissen_physik/
python3 /deepspeech-polyglot/corpora/prepare_datasets.py --youtube_dir /data_original/${LANGUAGE}/musstewissen_chemie/ /data_prepared/${LANGUAGE}/musstewissen_chemie/
python3 /deepspeech-polyglot/corpora/prepare_datasets.py --youtube_dir /data_original/${LANGUAGE}/pulsreportage/ /data_prepared/${LANGUAGE}/pulsreportage/
python3 /deepspeech-polyglot/corpora/prepare_datasets.py --youtube_dir /data_original/${LANGUAGE}/terrax/ /data_prepared/${LANGUAGE}/terrax/
python3 /deepspeech-polyglot/corpora/prepare_datasets.py --youtube_dir /data_original/${LANGUAGE}/ykollektiv/ /data_prepared/${LANGUAGE}/ykollektiv/
python3 /deepspeech-polyglot/preprocessing/prepare_data.py --zamia_speech /data_original/${LANGUAGE}/zamia_speech/ /data_prepared/${LANGUAGE}/zamia_speech/

# To split tuda into the correct train, dev and test splits run:
# (you will have to rename the [train/dev/test]_s.csv files before combining them with other datasets)
python3 deepspeech-polyglot/preprocessing/split_dataset.py /data_prepared/de/tuda/all.csv --tuda --file_appendix _s
# To split common-voice into the correct train, dev and test splits run:
python3 deepspeech-polyglot/preprocessing/split_dataset.py /data_prepared/${LANGUAGE}/common_voice/all.csv --common_voice --common_voice_org /data_original/${LANGUAGE}/common_voice/${LANGUAGE}/ --file_appendix _s
```

Preparation times for german datasets using Intel i7-8700K:

- voxforge: some seconds
- tuda: some minutes
- mailabs: ~20min
- common_voice: ~12h
- swc: ~6h

<br>

Some datasets are in the wrong format, convert them from DeepSpeech to corcua format like this:

```bash
python3 /deepspeech-polyglot/preprocessing/convert_ds2cc.py --source_path "/data_prepared/de/tuda/" \
  --target_path "/data_prepared/de/tuda2/" --train "train_s.csv" --dev "dev_s.csv" --test "test_s.csv"
```

Other commands:

```bash
export LANGUAGE="de"

# To replace non alphabet characters and clean the files run (repeat for all 3 csv files, but don't clean the test file):
python3 /deepspeech-polyglot/preprocessing/dataset_operations.py "/data_prepared/en/librispeech/train-all.csv" \
  "/data_prepared/en/librispeech/train-all_azce.csv" --replace --clean --exclude

# Combine specific csv files:
python3 /deepspeech-polyglot/preprocessing/combine_datasets.py --file_output "/data_prepared/en/librispeech/train-all.csv" \
  --files "/data_prepared/en/librispeech/train-clean-100.csv /data_prepared/en/librispeech/train-clean-360.csv /data_prepared/en/librispeech/train-other-500.csv"
```

<br/>

#### Download and prepare noise data

Run in container:

```bash
cd /data_original/noise/

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
python /deepspeech-polyglot/preprocessing/normalize_noise_audio.py --from_dir /data_original/noise/ --to_dir /data_prepared/noise/ --max_sec 45

# Create csv files:
python3 /deepspeech-polyglot/preprocessing/noise_to_csv.py
python3 /deepspeech-polyglot/preprocessing/split_dataset.py /data_prepared/noise/all.csv  --split "70|15|15"
```
