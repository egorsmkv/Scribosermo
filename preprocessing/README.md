# Preparing voice data

Downloading and preprocessing the datasets. Run all scripts inside the container. \
Try preparation and training with a small dataset first, before you start to download all the others.

<br>

## Datasets

**German (de):**
[BAS-Formtask](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/FORMTASK/FORMTASK.2.php) (~18h),
[BAS-Sprecherinnen](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/SprecherInnen/SprecherInnen.1.php) (~2h),
[Common Voice](https://voice.mozilla.org/) (~701h),
[Common Voice Single Words](https://voice.mozilla.org/) (~9h, included in the main dataset),
[CSS10](https://www.kaggle.com/bryanpark/german-single-speaker-speech-dataset) (~16h),
GoogleWavenet (~165h, artificial training data generated with the google text to speech service),
Gothic (~39h, extracted from Gothic 1-3 games),
[Kurzgesagt](https://www.youtube.com/c/KurzgesagtDE/videos) (~8h),
[LinguaLibre](https://lingualibre.org/wiki/LinguaLibre:Main_Page) (~4h),
[M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) (~234h),
MussteWissen [Deutsch](https://www.youtube.com/c/musstewissenDeutsch/videos) [Mathe](https://www.youtube.com/c/musstewissenMathe/videos) [Physik](https://www.youtube.com/c/musstewissenPhysik/videos) [Chemie](https://www.youtube.com/c/musstewissenChemie/videos) (~11h),
[PULS-Reportage](https://www.youtube.com/puls/videos) (~12h),
[Spoken Wikipedia Corpora (SWC)](https://nats.gitlab.io/swc/) (~248h),
[Tatoeba](https://tatoeba.org/deu/sentences/search?query=&from=deu&to=und&user=&orphans=no&unapproved=no&has_audio=yes&tags=&list=&native=&trans_filter=limit&trans_to=und&trans_link=&trans_user=&trans_orphan=&trans_unapproved=&trans_has_audio=&sort_reverse=&sort=relevance) (~7h),
[TerraX](https://www.youtube.com/c/terra-x/videos) (~38h),
[German Distant Speech Corpus (TUDA-De)](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/acoustic-models.html) (~185h),
[Voxforge](http://www.voxforge.org/home/forums/other-languages/german/open-speech-data-corpus-for-german) (~32h),
[Y-Kollektiv](https://www.youtube.com/c/ykollektiv/videos) (~52h),
[Zamia-Speech](https://goofy.zamia.org/zamia-speech/corpora/zamia_de/) (~19h),

**English (en):**
[LibriSpeech](http://www.openslr.org/11) (~982h),

**Spanish (es):**
[Common Voice](https://voice.mozilla.org/) (~390h),
[CSS10](https://www.kaggle.com/bryanpark/spanish-single-speaker-speech-dataset) (~24h),
[LinguaLibre](https://lingualibre.org/wiki/LinguaLibre:Main_Page) (~1h),
[M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) (~108h),
[Tatoeba](https://tatoeba.org/spa/sentences/search?query=&from=spa&to=und&user=&orphans=no&unapproved=no&has_audio=yes&tags=&list=&native=&trans_filter=limit&trans_to=und&trans_link=&trans_user=&trans_orphan=&trans_unapproved=&trans_has_audio=&sort_reverse=&sort=relevance) (~59h),
[Voxforge](http://www.voxforge.org/home/) (~52h),

**French (fr):**
[Common Voice](https://voice.mozilla.org/) (~546h),
[CSS10](https://www.kaggle.com/bryanpark/french-single-speaker-speech-dataset) (~19h),
[LinguaLibre](https://lingualibre.org/wiki/LinguaLibre:Main_Page) (~40h),
[M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) (~184h),
[Tatoeba](https://tatoeba.org/fra/sentences/search?query=&from=fra&to=und&user=&orphans=no&unapproved=no&has_audio=yes&tags=&list=&native=&trans_filter=limit&trans_to=und&trans_link=&trans_user=&trans_orphan=&trans_unapproved=&trans_has_audio=&sort_reverse=&sort=relevance) (~2h),
[Voxforge](http://www.voxforge.org/home/) (~37h),

**Italian (it):**
[Common Voice](https://voice.mozilla.org/) (~149h),
[LinguaLibre](https://lingualibre.org/wiki/LinguaLibre:Main_Page) (~0h),
[M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) (~127h),
[Voxforge](http://www.voxforge.org/home/) (~20h),

**Polish (pl):**
[Common Voice](https://voice.mozilla.org/) (~113h),
[LinguaLibre](https://lingualibre.org/wiki/LinguaLibre:Main_Page) (~2h),
[M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) (~54h),

**Noise:**
[Freesound Dataset Kaggle 2019](https://zenodo.org/record/3612637#.Xjq7OuEo9rk) (~103h),
[RNNoise](https://people.xiph.org/~jm/demo/rnnoise/) (~44h),
[Zamia-Noise](http://goofy.zamia.org/zamia-speech/corpora/noise.tar.xz) (~5h),

<br>

## Setup

The training datasets have to be in `tab` separated `.csv` format,
containing at least the columns `filepath` (absolute), `duration` (seconds), `text`.
All other columns will be ignored automatically. \
Audio files have to be in `.wav` format, with 16kHz recording rate and a single channel only.

<br>

#### Download datasets

A lot of datasets have to be downloaded by hand, but for a few there are download scripts.

Download the German youtube playlists like this:

```bash
python3 /deepspeech-polyglot/preprocessing/download_playlists.py --target_path "/data_original/de/" [InsertDatasetHere]

# Choose one of those datasets
--kurzgesagt --musstewissen --pulsreportage --terrax --ykollektiv
```

You can download some datasets with _corcua_ (see [corcua's readme](https://gitlab.com/Jaco-Assistant/corcua#usage-examples) for special arguments):

```bash
python3 -c 'from corcua import downloaders; downloaders.mls.Downloader().download_dataset(path="/data_original/de/MLS/", overwrite=True, args={"language": "de"})'
```

For the datasets from _audiomate_ use those commands:

```bash
python3 -c 'from audiomate.corpus import io; io.SWCDownloader(lang="de").download("/data_original/de/swc/")'
python3 -c 'from audiomate.corpus import io; io.TatoebaDownloader(include_languages=["de"]).download("/data_original/de/tatoeba/")'
python3 -c 'from audiomate.corpus import io; io.TudaDownloader().download("/data_original/de/tuda"/)'
```

<br/>

#### Prepare datasets

Depending on the dataset size, this step may take some hours.

Prepare datasets with _Corcua_ like this (see [corcua's readme](https://gitlab.com/Jaco-Assistant/corcua#usage-examples) for special arguments):

```bash
python3 -c 'from corcua import readers, writers; \
  ds = readers.mls.Reader().load_dataset({"path": "/data_original/de/MLS/mls_german_opus/"}); \
  writers.base_writer.Writer().save_dataset(ds, path="/data_prepared/de/MLS/", sample_rate=16000, overwrite=True);'
```

And datasets from _Audiomate_:

```bash
python3 -c 'import audiomate; \
  ds = audiomate.Corpus.load("/data_original/de/swc/", reader="swc"); \
  io.MozillaDeepSpeechWriter().save(ds, "/data_prepared/de/swc/");'
```

<br>

Split _tuda_ dataset into the correct partitions:

```bash
python3 deepspeech-polyglot/preprocessing/split_dataset.py /data_prepared/de/tuda/all.csv --tuda --file_appendix _s
```

Some datasets (those downloaded with audiomate for example) are in the wrong format, you can convert them from _DeepSpeech_ to _corcua_ format like this:

```bash
python3 /deepspeech-polyglot/preprocessing/convert_ds2cc.py --source_path "/data_prepared/de/tuda/" \
  --target_path "/data_prepared/de/tuda2/" --train "train_s.csv" --dev "dev_s.csv" --test "test_s.csv"

# Remove "/DeepSpeech" path prefix
sed 's/\/DeepSpeech//g' train.csv > train2.csv
```

Replace non alphabet characters and remove some audio files:

```bash
export LANGUAGE="de"

# Repeat for all 3 csv files, but don't clean the test file:
python3 /deepspeech-polyglot/preprocessing/dataset_operations.py "/data_prepared/en/librispeech/train-all.csv" \
  "/data_prepared/en/librispeech/train-all_azce.csv" --replace --exclude --clean

```

Combine specific csv files:

```bash
python3 /deepspeech-polyglot/preprocessing/combine_datasets.py --file_output "/data_prepared/en/librispeech/train-all.csv" \
  --files "/data_prepared/en/librispeech/train-clean-100.csv /data_prepared/en/librispeech/train-clean-360.csv /data_prepared/en/librispeech/train-other-500.csv"
```

<br/>

#### Download and prepare noise data

Note: Currently the noise files can't be used for automatic augmentation.
So if you want to train with the new network implementations, you can skip this step.

Get data:

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
```

Prepare it:

```bash
# Normalize all the audio files (run with python2):
python /deepspeech-polyglot/preprocessing/normalize_noise_audio.py --from_dir /data_original/noise/ --to_dir /data_prepared/noise/ --max_sec 30

# Create csv files:
python3 /deepspeech-polyglot/preprocessing/noise_to_csv.py
python3 /deepspeech-polyglot/preprocessing/split_dataset.py /data_prepared/noise/all.csv  --split "70|15|15"
```
