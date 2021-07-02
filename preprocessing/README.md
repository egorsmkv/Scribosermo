# Preparing voice data

Downloading and preprocessing the datasets. Run all scripts inside the container. \
Try preparation and training with a small dataset first, before you start to download all the others.

<br>

## Datasets

**German (de):**
[Alcohol Language Corpus](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/ALC/ALC.4.php) (~48h),
[BAS-Formtask](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/FORMTASK/FORMTASK.2.php) (~18h),
[BAS-Sprecherinnen](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/SprecherInnen/SprecherInnen.1.php) (~2h),
[Brothers](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/BROTHERS/BROTHERS.2.php) (~7h),
[Common Voice](https://voice.mozilla.org/) (~777h),
[Common Voice Single Words](https://voice.mozilla.org/) (~9h, included in the main dataset),
[CSS10](https://www.kaggle.com/bryanpark/german-single-speaker-speech-dataset) (~16h),
GoogleWavenet (~165h, artificial training data generated with the google text to speech service),
Gothic (~39h, extracted from Gothic 1-3 games),
[Guild2-Renaissance](https://www.gog.com/game/the_guild_2_renaissance) (~11h),
[Hempel](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/HEMPEL/HEMPEL.4.php) (~25h),
[Kurzgesagt](https://www.youtube.com/c/KurzgesagtDE/videos) (~9h),
[LinguaLibre](https://lingualibre.org/wiki/LinguaLibre:Main_Page) (~4h),
[M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) (~234h),
[Multilingual LibriSpeech](http://www.openslr.org/94/) (~1995h),
[Multilingual TEDx](http://www.openslr.org/100/) (~14h),
MussteWissen [Deutsch](https://www.youtube.com/c/musstewissenDeutsch/videos) [Mathe](https://www.youtube.com/c/musstewissenMathe/videos) [Physik](https://www.youtube.com/c/musstewissenPhysik/videos) [Chemie](https://www.youtube.com/c/musstewissenChemie/videos) (~11h),
[PhattSessionz](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/PHATTSESSIONZ/PHATTSESSIONZ.2.php) (~238h),
[PhoneDat 1](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/PD1/PD1.3.php) (~21h),
[PULS-Reportage](https://www.youtube.com/puls/videos) (~16h),
[Regional Variants of German](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/RVG1_CLARIN/RVG1_CLARIN.3.php) (~129h),
[RVG - Juveniles](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/RVG-J/RVG-J.2.php) (~49h),
[SC10](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/SC10/SC10.4.php) (~6h),
[Smartweb Handheld Corpus](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/SHC/SHC.2.php) (~29h),
[SI100](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/SI100/SI100.2.php) (~36h),
[Skyrim Legacy+DLCs](https://store.steampowered.com/app/72850/The_Elder_Scrolls_V_Skyrim/) (~89h),
[Smartweb Motorbike Corpus](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/SMC/SMC.2.php) (~6h),
[Spoken Wikipedia Corpora](https://nats.gitlab.io/swc/) (~248h),
[Tatoeba](https://tatoeba.org/deu/sentences/search?query=&from=deu&to=und&user=&orphans=no&unapproved=no&has_audio=yes&tags=&list=&native=&trans_filter=limit&trans_to=und&trans_link=&trans_user=&trans_orphan=&trans_unapproved=&trans_has_audio=&sort_reverse=&sort=relevance) (~8h),
[Thorsten](http://www.openslr.org/95/) (~23h),
[TerraX](https://www.youtube.com/c/terra-x/videos) (~48h),
[TUDA](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/acoustic-models.html) (~185h),
[Verbmobil 1](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/VM1/VM1.3.php) (~34h),
[Verbmobil 2](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/VM2/VM2.3.php) (~22h),
[Voxforge](http://www.voxforge.org/home/forums/other-languages/german/open-speech-data-corpus-for-german) (~33h),
[WaSeP](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/WaSeP/WaSeP.2.php) (~3h),
[Witcher3-GOTY](https://www.gog.com/game/the_witcher_3_wild_hunt_game_of_the_year_edition) (~44h),
[Y-Kollektiv](https://www.youtube.com/c/ykollektiv/videos) (~58h),
[Zamia-Speech](https://goofy.zamia.org/zamia-speech/corpora/zamia_de/) (~19h),
[ZipTel](https://clarin.phonetik.uni-muenchen.de/BASRepository/Public/Corpora/ZIPTEL/ZIPTEL.3.php) (~13h)

**English (en):**
[LibriSpeech](http://www.openslr.org/11) (~982h)

**Spanish (es):**
[Common Voice](https://voice.mozilla.org/) (~331h),
[CSS10](https://www.kaggle.com/bryanpark/spanish-single-speaker-speech-dataset) (~24h),
[LibriVox-Spanish](https://www.kaggle.com/carlfm01/120h-spanish-speech) (~120h),
[LinguaLibre](https://lingualibre.org/wiki/LinguaLibre:Main_Page) (~1h),
[M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) (~109h),
[mTEDx](http://www.openslr.org/100/) (~185h),
[Tatoeba](https://tatoeba.org/spa/sentences/search?query=&from=spa&to=und&user=&orphans=no&unapproved=no&has_audio=yes&tags=&list=&native=&trans_filter=limit&trans_to=und&trans_link=&trans_user=&trans_orphan=&trans_unapproved=&trans_has_audio=&sort_reverse=&sort=relevance) (~60h),
[Voxforge](http://www.voxforge.org/home/) (~52h)

**French (fr):**
[Common Voice](https://voice.mozilla.org/) (~617h),
[CSS10](https://www.kaggle.com/bryanpark/french-single-speaker-speech-dataset) (~19h),
[LinguaLibre](https://lingualibre.org/wiki/LinguaLibre:Main_Page) (~45h),
[M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) (~184h),
[mTEDx](http://www.openslr.org/100/) (~183h),
[Tatoeba](https://tatoeba.org/fra/sentences/search?query=&from=fra&to=und&user=&orphans=no&unapproved=no&has_audio=yes&tags=&list=&native=&trans_filter=limit&trans_to=und&trans_link=&trans_user=&trans_orphan=&trans_unapproved=&trans_has_audio=&sort_reverse=&sort=relevance) (~2h),
[Voxforge](http://www.voxforge.org/home/) (~37h)

**Italian (it):**
[Common Voice](https://voice.mozilla.org/) (~157h),
[LinguaLibre](https://lingualibre.org/wiki/LinguaLibre:Main_Page) (~1h),
[M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) (~128h),
[mTEDx](http://www.openslr.org/100/) (~106h),
[Voxforge](http://www.voxforge.org/home/) (~20h)

**Polish (pl):**
[Common Voice](https://voice.mozilla.org/) (~113h),
[LinguaLibre](https://lingualibre.org/wiki/LinguaLibre:Main_Page) (~2h),
[M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) (~54h)

**Noise:**
[Freesound Dataset Kaggle 2019](https://zenodo.org/record/3612637#.Xjq7OuEo9rk) (~103h),
[RNNoise](https://people.xiph.org/~jm/demo/rnnoise/) (~44h),
[Zamia-Noise](http://goofy.zamia.org/zamia-speech/corpora/noise.tar.xz) (~5h)

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
python3 /Scribosermo/preprocessing/download_playlists.py --target_path "/data_original/de/" [InsertDatasetHere]

# Choose one of those datasets
--kurzgesagt --musstewissen --pulsreportage --terrax --ykollektiv
```

You can download some datasets with _corcua_ (see [corcua's readme](https://gitlab.com/Jaco-Assistant/corcua#usage-examples) for special arguments):

```bash
python3 -c 'from corcua import downloaders; downloaders.mls.Downloader().download_dataset(path="/data_original/de/MLS/", overwrite=True, args={"language": "de"}); print("FINISHED");'
```

For the datasets from _audiomate_ use those commands:

```bash
python3 -c 'from audiomate.corpus import io; io.SWCDownloader(lang="de").download("/data_original/de/swc/"); print("FINISHED");'
python3 -c 'from audiomate.corpus import io; io.TudaDownloader().download("/data_original/de/tuda"/); print("FINISHED");'
```

<br/>

#### Prepare datasets

Depending on the dataset size, this step may take some hours.

Prepare datasets with _corcua_ like this (see [corcua's readme](https://gitlab.com/Jaco-Assistant/corcua#usage-examples) for special arguments):

```bash
python3 -c 'from corcua import readers, writers; \
  ds = readers.mls.Reader().load_dataset({"path": "/data_original/de/MLS/mls_german_opus/"}); \
  writers.base_writer.Writer().save_dataset(ds, path="/data_prepared/de/MLS/", sample_rate=16000, overwrite=True); print("FINISHED");'
```

And datasets from _audiomate_:

```bash
python3 -c 'import audiomate; from audiomate.corpus import io; \
  ds = audiomate.Corpus.load("/data_original/de/swc/", reader="swc"); \
  io.MozillaDeepSpeechWriter().save(ds, "/data_prepared/de/swc/"); print("FINISHED");'
```

<br>

Split _tuda_ dataset into the correct partitions:

```bash
python3 Scribosermo/preprocessing/split_dataset.py /data_prepared/de/tuda/all.csv --tuda --file_appendix _s
```

Some datasets (those downloaded with audiomate for example) are in _DeepSpeech_ format, you can convert them to _corcua_ format like this:

```bash
python3 /Scribosermo/preprocessing/convert_ds2cc.py --source_path "/data_prepared/de/tuda/" \
  --target_path "/data_prepared/de/tuda2/" --train "train_s.csv" --dev "dev_s.csv" --test "test_s.csv" --remove_text_commas

# Remove "/DeepSpeech" path prefix (old directory structure)
sed 's/\/DeepSpeech//g' train.csv > train2.csv
```

Replace non alphabet characters and clean out some audio files:

```bash
export LANGUAGE="de"

# Repeat for all 3 csv files, but don't clean the test file:
python3 /Scribosermo/preprocessing/dataset_operations.py "/data_prepared/${LANGUAGE}/common_voice/train.csv" \
  "/data_prepared/${LANGUAGE}/common_voice/train_azce.csv" --replace --exclude --clean
```

Combine specific csv files: \
(you can either use `--files_str` like shown below or `--files_txt` with a path to a text file containing one dataset per line)

```bash
python3 /Scribosermo/preprocessing/combine_datasets.py --file_output "/data_prepared/en/librispeech/train-all.csv" \
  --files_str "/data_prepared/en/librispeech/train-clean-100.csv /data_prepared/en/librispeech/train-clean-360.csv /data_prepared/en/librispeech/train-other-500.csv"
```

<br/>

#### Download and prepare noise data

Note: Currently the noise files can't be used for automatic augmentation. This was a feature used in DeepSpeech.
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
python /Scribosermo/preprocessing/normalize_noise_audio.py --from_dir /data_original/noise/ --to_dir /data_prepared/noise/ --max_sec 30

# Create csv files:
python3 /Scribosermo/preprocessing/noise_to_csv.py
python3 /Scribosermo/preprocessing/split_dataset.py /data_prepared/noise/all.csv  --split "70|15|15"
```
