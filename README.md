# DeepSpeech German

_This project is build upon the paper [German End-to-end Speech Recognition based on DeepSpeech](https://www.researchgate.net/publication/336532830_German_End-to-end_Speech_Recognition_based_on_DeepSpeech)._
_Original paper code can be found [here](https://github.com/AASHISHAG/deepspeech-german)._

This project aims to develop a working Speech to Text module using [Mozilla DeepSpeech](https://github.com/mozilla/DeepSpeech), which can be used for any Audio processing pipeline. [Mozillla DeepSpeech](https://github.com/mozilla/DeepSpeech) is a state-of-the-art open-source automatic speech recognition (ASR) toolkit. DeepSpeech is using a model trained by machine learning techniques based on [Baidu's Deep Speech](https://gigaom2.files.wordpress.com/2014/12/deep_speech3_12_17.pdf) research paper. Project DeepSpeech uses Google's TensorFlow to make the implementation easier.

<p align="center">
    <img src="media/deep_speech_architecture.png" align="center" title="DeepSpeech Graph" />
</p>

## Usage

#### General infos

File structure will look as follows:

```
my_deepspeech_folder
    checkpoints
    data_original
    data_prepared
    DeepSpeech
    deepspeech-german    <- This repositiory
```

Clone DeepSpeech and build docker container:

```
git clone https://github.com/mozilla/DeepSpeech.git
# or
git clone https://github.com/DanBmh/DeepSpeech.git

docker build -t mozilla_deep_speech DeepSpeech/
```

Build and run our docker container:
```
docker build -t deep_speech_german deepspeech-german/

./deepspeech-german/run_container.sh
```

#### Download and prepare voice data

* [German Distant Speech Corpus (TUDA-De)](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/acoustic-models.html) ~185h
* [Mozilla Common Voice](https://voice.mozilla.org/) ~506h
* [Voxforge](http://www.voxforge.org/home/forums/other-languages/german/open-speech-data-corpus-for-german) ~32h
* [Spoken Wikipedia Corpora (SWC)](https://nats.gitlab.io/swc/) ~248h
* [M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) ~234h
* GoogleWavenet ~165h, artificial training data generated with the google text to speech service
* [Tatoeba](https://tatoeba.org/deu/sentences/search?query=&from=deu&to=und&user=&orphans=no&unapproved=no&has_audio=yes&tags=&list=&native=&trans_filter=limit&trans_to=und&trans_link=&trans_user=&trans_orphan=&trans_unapproved=&trans_has_audio=&sort_reverse=&sort=relevance) ~7h

* Not used: [Forschergeist](https://forschergeist.de/archiv/) ~100-150h, no data pipeline existing
* Not used: [Zamia](https://goofy.zamia.org/zamia-speech/corpora/zamia_de/) ~18h
* Not used: [Zamia-Noise](http://goofy.zamia.org/zamia-speech/corpora/noise.tar.xz) ~?h
* Noise data: [Freesound Dataset Kaggle 2019](https://zenodo.org/record/3612637#.Xjq7OuEo9rk) ~103h
* Noise data: [RNNoise](https://people.xiph.org/~jm/demo/rnnoise/) ~44h

<br/>

Download datasets (Run in docker container):
```
python3 deepspeech-german/data/download_data.py --tuda data_original/
python3 deepspeech-german/data/download_data.py --voxforge data_original/
python3 deepspeech-german/data/download_data.py --mailabs data_original/
python3 deepspeech-german/data/download_data.py --swc data_original/
python3 deepspeech-german/data/download_data.py --tatoeba data_original/
```

Download common voice dataset: https://voice.mozilla.org/en/datasets \
Extract and move it to datasets directory (data_original/common_voice/)

<br/>

Prepare datasets, this may take some time (Run in docker container):
```
# Prepare the datasets one by one first to ensure everything is working:

./deepspeech-german/pre-processing/run_to_utf_8.sh "/DeepSpeech/data_original/voxforge/*/etc/prompts-original"
python3 deepspeech-german/pre-processing/prepare_data.py --voxforge data_original/voxforge/  data_prepared/voxforge/

python3 deepspeech-german/pre-processing/prepare_data.py --tuda data_original/tuda/  data_prepared/tuda/
python3 deepspeech-german/pre-processing/prepare_data.py --common_voice data_original/common_voice/  data_prepared/common_voice/
python3 deepspeech-german/pre-processing/prepare_data.py --mailabs data_original/mailabs/  data_prepared/mailabs/
python3 deepspeech-german/pre-processing/prepare_data.py --swc data_original/swc/  data_prepared/swc/
python3 deepspeech-german/pre-processing/prepare_data.py --tatoeba data_original/tatoeba/  data_prepared/tatoeba/


# To combine multiple datasets run the command as follows:
python3 deepspeech-german/pre-processing/prepare_data.py --tuda data_original/tuda/ --voxforge data_original/voxforge/  data_prepared/tuda_voxforge/

# Or, which is much faster, but only combining train, dev, test and all csv files, run:
python3 deepspeech-german/data/combine_datasets.py data_prepared/ --tuda --voxforge

# Or to combine specific csv files:
python3 deepspeech-german/data/combine_datasets.py "" --files_output data_prepared/tuda-voxforge-swc-mailabs-common_voice/train_mix.csv --files "data_prepared/tuda/train.csv data_prepared/voxforge/all.csv data_prepared/swc/all.csv data_prepared/mailabs/all.csv data_prepared/common_voice/train.csv"


# To shuffle and replace "äöü" characters and clean the files run (for all 3 csv files):
python3 /DeepSpeech/deepspeech-german/data/dataset_operations.py /DeepSpeech/data_prepared/tuda-voxforge/train.csv /DeepSpeech/data_prepared/tuda-voxforge/train_azce.csv --replace --shuffle --clean --exclude


# To split tuda into the correct train, dev and test splits run: 
# (you will have to rename the [train/dev/test]_s.csv files before combining them with other datasets)
python3 deepspeech-german/data/split_dataset.py data_prepared/tuda/ --tuda --file_appendix _s
```

Preparation times using Intel i7-8700K:
* voxforge: some seconds
* tuda: some minutes
* mailabs: ~20min
* common_voice: ~12h
* swc: ~6h

<br/>

Download and extract noise data (You have to merge https://github.com/mozilla/DeepSpeech/pull/2622 for that, run in container):

```
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
wget https://people.xiph.org/~jm/demo/rnnoise/rnnoise_contributions.tar.gz
tar -xvzf rnnoise_contributions.tar.gz
rm rnnoise_contributions.tar.gz

# Normalize all the audio files:
cd /DeepSpeech/
python deepspeech-german/data/normalize_noise_audio.py --from_dir data_original/noise/ --to_dir data_prepared/noise/ --max_sec 45
```

#### Create the language model

Download and prepare the open-source [German Speech Corpus](http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/German_sentences_8mil_filtered_maryfied.txt.gz):
```
wget http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/German_sentences_8mil_filtered_maryfied.txt.gz -O data_original/sentences.txt.gz
gzip -d data_original/sentences.txt.gz

python3 deepspeech-german/pre-processing/prepare_vocab.py data_original/sentences.txt data_prepared/clean_vocab_az.txt --replace_umlauts
```

Generate scorer (Run in docker container):
```bash
mkdir data_prepared/lm/

python3 /DeepSpeech/data/lm/generate_lm.py --input_txt /DeepSpeech/data_prepared/clean_vocab_az.txt --output_dir /DeepSpeech/data_prepared/lm/ --top_k 500000 --kenlm_bins /DeepSpeech/native_client/kenlm/build/bin/ --arpa_order 5 --max_arpa_memory "85%" --arpa_prune "0|0|1" --binary_a_bits 255 --binary_q_bits 8 --binary_type trie

python3 /DeepSpeech/data/lm/generate_package.py --alphabet /DeepSpeech/deepspeech-german/data/alphabet_az.txt --lm /DeepSpeech/data_prepared/lm/lm.binary --vocab /DeepSpeech/data_prepared/lm/vocab-500000.txt --package /DeepSpeech/data_prepared/lm/kenlm_az.scorer --default_alpha 0.75 --default_beta 1.85
```

#### Fix some issues

For me only training with voxforge worked at first. With tuda dataset I got an error: \
"Invalid argument: Not enough time for target transition sequence"

To fix it you have to follow this [solution](https://github.com/mozilla/DeepSpeech/issues/1629#issuecomment-427423707):
```
# Add the parameter "ignore_longer_outputs_than_inputs=True" in DeepSpeech.py (~ line 231)

# Compute the CTC loss using TensorFlow's `ctc_loss`
total_loss = tfv1.nn.ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_seq_len, ignore_longer_outputs_than_inputs=True)
```

<br/>

This will result in another error after some training steps: \
"Invalid argument: WAV data chunk '[Some strange symbol here]"

Just ignore this in the train steps:
```
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

```
# Below this lines (DeepSpeech.py ~ line 620):

problem_files = [f.decode('utf8') for f in problem_files[..., 0]]
log_error('The following files caused an infinite (or NaN) '
          'loss: {}'.format(','.join(problem_files)))


# Add the following to save the files to excluded_files.json and stop training

sys.path.append("/DeepSpeech/deepspeech-german/training/")
from filter_invalid_files import add_files_to_excluded
add_files_to_excluded(problem_files)
sys.exit(1)
```

#### Training

Download pretrained deepspeech checkpoints.

```
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.6.0/deepspeech-0.6.0-checkpoint.tar.gz -P checkpoints/
tar xvfz checkpoints/deepspeech-0.6.0-checkpoint.tar.gz -C checkpoints/
rm checkpoints/deepspeech-0.6.0-checkpoint.tar.gz
```

Adjust the parameters to your needs (Run in docker container):

```
# Delete old model files:
rm -rf /root/.local/share/deepspeech/summaries && rm -rf /root/.local/share/deepspeech/checkpoints


# Run training:
python3 DeepSpeech.py --train_files data_prepared/voxforge/train.csv --dev_files data_prepared/voxforge/dev.csv --test_files data_prepared/voxforge/test.csv \
--alphabet_config_path deepspeech-german/data/alphabet.txt --lm_trie_path data_prepared/trie --lm_binary_path data_prepared/lm.binary --test_batch_size 48 --train_batch_size 48 --dev_batch_size 48 \
--epochs 75 --learning_rate 0.0005 --dropout_rate 0.40 --export_dir deepspeech-german/models --use_allow_growth --use_cudnn_rnn 

# Or adjust the train.sh file and run a training using the english checkpoint:
/bin/bash /DeepSpeech/deepspeech-german/training/train.sh /DeepSpeech/checkpoints/voxforge/ /DeepSpeech/data_prepared/voxforge/train_azce.csv /DeepSpeech/data_prepared/voxforge/dev_azce.csv /DeepSpeech/data_prepared/voxforge/test_azce.csv 1 /DeepSpeech/checkpoints/deepspeech-0.6.0-checkpoint/

# Or to run a cycled training as described in the paper, run:
python3 /DeepSpeech/deepspeech-german/training/cycled_training.py /DeepSpeech/checkpoints/voxforge/ /DeepSpeech/data_prepared/ _azce --voxforge


# Run test only (The use_allow_growth flag fixes "cuDNN failed to initialize" error):
# Don't forget to add the noise augmentation flags if testing with noise
python3 /DeepSpeech/DeepSpeech.py --test_files /DeepSpeech/data_prepared/voxforge/test_azce.csv --checkpoint_dir /DeepSpeech/checkpoints/voxforge/ --scorer_path /DeepSpeech/data_prepared/lm/kenlm_az.scorer --alphabet_config_path /DeepSpeech/deepspeech-german/data/alphabet_az.txt --test_batch_size 36 --use_allow_growth
```

Training time for voxforge on 2x Nvidia 1080Ti using batch size of 48 is about 01:45min per epoch. Training until early stop took 22min for 10 epochs. 

One epoch in tuda with batch size of 12 on single gpu needs about 1:15h. With both gpus it takes about 26min. For 10 cycled training with early stops it took about 15h.

One epoch in mailabs with batch size of 24/12/12 needs about 19min, testing about 21 min. 

One epoch in swc with batch size of 12/12/12 needs about 1:08h, testing about 17 min.

One epoch with all datasets and batch size of 12 needs about 2:50h, testing about 1:30h. Training until early stop took 37h for 11 epochs.

One epoch with all datasets and only Tuda + CommonVoice as testset needs about 3:30h. Training for 55 epochs took 8d 6h, testing about 1h.

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

| Dataset | Additional Infos | Losses | Result |
|---------|------------------|--------|--------|
| Tuda + CommonVoice | used newer CommonVoice version, there may be overlaps between test and training data because of random splitting | Test: 105.747589 | WER: 0.683802, CER: 0.386331 |
| Tuda | correct tuda test split, there may be overlaps between test and training data because of random splitting | Test: 402.696991 | WER: 0.785655, CER: 0.428786 |

<br/>

#### This repo

Some results with a old code version (Default dropout is 0.4, learning rate 0.0005): 

| Dataset | Additional Infos | Result |
|---------|------------------|--------|
| Voxforge | | WER: 0.676611, CER: 0.403916, loss: 82.185226 |
| Voxforge | with augmentation | WER: 0.624573, CER: 0.348618, loss: 74.403786 |
| Voxforge | without "äöü" | WER: 0.646702, CER: 0.364471, loss: 82.567413 |
| Voxforge | cleaned data, without "äöü" | WER: 0.634828, CER: 0.353037, loss: 81.905258 |
| Voxforge | above checkpoint, tested on not cleaned data | WER: 0.634556, CER: 0.352879, loss: 81.849220 |
| Voxforge | checkpoint from english deepspeech, without "äöü" | WER: 0.394064, CER: 0.190184, loss: 49.066357 |
| Voxforge | checkpoint from english deepspeech, with augmentation, without "äöü", dropout 0.25, learning rate 0.0001 | WER: 0.338685, CER: 0.150972, loss: 42.031754 |
| Voxforge | reduce learning rate on plateau, with noise and standard augmentation, checkpoint from english deepspeech, cleaned data, without "äöü", dropout 0.25, learning rate 0.0001, batch size 48 | WER: 0.320507, CER: 0.131948, loss: 39.923031 |
| Voxforge | above with learning rate 0.00001 | WER: 0.350903, CER: 0.147837, loss: 43.451263 |
| Voxforge | above with learning rate 0.001 | WER: 0.518670, CER: 0.252510, loss: 62.927200 |
| Tuda + Voxforge | without "äöü", checkpoint from english deepspeech, cleaned train and dev data | WER: 0.740130, CER: 0.462036, loss: 156.115921 |
| Tuda + Voxforge | first Tuda then Voxforge, without "äöü", cleaned train and dev data, dropout 0.25, learning rate 0.0001 | WER: 0.653841, CER: 0.384577, loss: 159.509476 |
| Tuda + Voxforge + SWC + Mailabs + CommonVoice | checkpoint from english deepspeech, with augmentation, without "äöü", cleaned data, dropout 0.25, learning rate 0.0001 | WER: 0.306061, CER: 0.151266, loss: 33.218510 |

<br/>

Some results with some older code version: \
(Default values: batch size 12, dropout 0.25, learning rate 0.0001, without "äöü", cleaned data , checkpoint from english deepspeech, early stopping, reduce learning rate on plateau, evaluation with scorer and top-500k words)

| Dataset | Additional Infos | Losses | Training epochs of best model | Result |
|---------|------------------|--------|-------------------------------|--------|
| Voxforge | batch size 36 | Test: 46.309738, Validation: 50.323496 | 12 | WER: 0.343841, CER: 0.134452 |
| Voxforge | above checkpoint tested with cocktail party augmentation | Test: 118.516922 | | WER: 0.689503, CER: 0.359209 |
| Voxforge | like above, cocktail party augmentation with same dataset | Test: 53.604279, Validation: 55.484096 | 22 | WER: 0.426425, CER: 0.212265 |
| Voxforge | above checkpoint tested without cocktail party augmentation | Test: 63.336746 | | WER: 0.431053, CER: 0.249465 |
| Tuda + Voxforge + SWC + Mailabs + CommonVoice  | test only with Tuda + CommonVoice others completely for training, language model with training transcriptions, with augmentation | Test: 29.363405, Validation: 23.509546 | 55 | WER: 0.190189, CER: 0.091737 |
| Tuda + Voxforge + SWC + Mailabs + CommonVoice  | above checkpoint tested with 3-gram language model | Test: 29.363405 | | WER: 0.199709, CER: 0.095318 |
| Tuda + Voxforge + SWC + Mailabs + CommonVoice  | above checkpoint tested on Tuda only | Test: 87.074394 | | WER: 0.378379, CER: 0.167380 |

<br/>

Some results with the current code version: \
(Default values: batch size 36, dropout 0.25, learning rate 0.0001, without "äöü", cleaned data , checkpoint from english deepspeech, early stopping, reduce learning rate on plateau, evaluation with scorer and top-500k words, data augmentation)

| Dataset | Additional Infos | Losses | Training epochs of best model | Result |
|---------|------------------|--------|-------------------------------|--------|
| Voxforge | training from scratch | Test: 79.124008, Validation: 81.982976 | 29 | WER: 0.603879, CER: 0.298139 |
| Voxforge | | Test: 44.312195, Validation: 47.915317 | 21 | WER: 0.343973, CER: 0.140119 |
| Voxforge | without reduce learning rate on plateau | Test: 46.160049, Validation: 48.926518 | 13 | WER: 0.367125, CER: 0.163931 |
| Voxforge | dropped last layer | Test: 49.844028, Validation: 52.722362 | 21 | WER: 0.389327, CER: 0.170563 |
| Voxforge | 5 cycled training | Test: 42.973358 | | WER: 0.353841, CER: 0.158554 |
||
| Tuda | training from scratch, correct train/dev/test splitting | Test: 149.653427, Validation: 137.645307 | 9 | WER: 0.606629, CER: 0.296630 |
| Tuda | correct train/dev/test splitting | Test: 103.179092, Validation: 132.243965 | 3 | WER: 0.436074, CER: 0.208135 |
| Tuda | dropped last layer, correct train/dev/test splitting | Test: 107.047821, Validation: 101.219325 | 6 | WER: 0.431361, CER: 0.195361 |
| Tuda | dropped last two layers, correct train/dev/test splitting | Test: 110.523621, Validation: 103.844562 | 5 | WER: 0.442421, CER: 0.204504 |
| Tuda | checkpoint from Voxforge with WER 0.344, correct train/dev/test splitting | Test: 100.846367, Validation: 95.410456 | 3 | WER: 0.416950, CER: 0.198177 |
| Tuda | 10 cycled training, checkpoint from Voxforge with WER 0.344, correct train/dev/test splitting | Test: 98.007607 | | WER: 0.410520, CER: 0.194091 |
| Tuda | random dataset splitting, checkpoint from Voxforge with WER 0.344 <br> Important Note: These results are not meaningful, because same transcriptions can occur in train and test set, only recorded with different microphones | Test: 23.322618, Validation: 23.094230 | 27 | WER: 0.090285, CER: 0.036212 |
||
| CommonVoice | checkpoint from Tuda with WER 0.417 | Test: 24.688297, Validation: 17.460029 | 35 | WER: 0.217124, CER: 0.085427 |
| CommonVoice | above tested with reduced testset where transcripts occurring in trainset were removed,  | Test: 33.376812 |  | WER: 0.211668, CER: 0.079157 |
| CommonVoice + GoogleWavenet | above tested with GoogleWavenet | Test: 17.653290 | | WER: 0.035807, CER: 0.007342 |
| CommonVoice | checkpoint from Voxforge with WER 0.344 | Test: 23.460932, Validation: 16.641201 | 35 | WER: 0.215584, CER: 0.084932 |
| CommonVoice | dropped last layer | Test: 24.480028, Validation: 17.505738 | 36 | WER: 0.220435, CER: 0.086921 |
||
| Tuda + GoogleWavenet | added GoogleWavenet to train data, dev/test from Tuda, checkpoint from Voxforge with WER 0.344 | Test: 95.555939,  Validation: 90.392490 | 3 | WER: 0.390291, CER: 0.178549 |
| Tuda + GoogleWavenet | GoogleWavenet as train data, dev/test from Tuda | Test: 346.486420,  Validation: 326.615474 | 0 | WER: 0.865683, CER: 0.517528 |
| Tuda + GoogleWavenet | GoogleWavenet as train/dev data, test from Tuda | Test: 477.049591,  Validation: 3.320163 | 23 | WER: 0.923973, CER: 0.601015 |
| Tuda + GoogleWavenet | above checkpoint tested with GoogleWavenet | Test: 3.406022 | | WER: 0.012919, CER: 0.001724 |
| Tuda + GoogleWavenet | checkpoint from english deepspeech tested with Tuda | Test: 402.102661 | | WER: 0.985554, CER: 0.752787 |
| Voxforge + GoogleWavenet | added all of GoogleWavenet to train data, dev/test from Voxforge | Test: 45.643063,  Validation: 49.620488 | 28 | WER: 0.349552, CER: 0.143108 |
| CommonVoice + GoogleWavenet | added all of GoogleWavenet to train data, dev/test from CommonVoice | Test: 25.029057,  Validation: 17.511973 | 35 | WER: 0.214689, CER: 0.084206 |
| CommonVoice + GoogleWavenet | above tested with reduced testset | Test: 34.191067 | | WER: 0.213164, CER: 0.079121 |

#### Language Model and Checkpoints

Scorer with training transcriptions: [Link](https://megastore.uni-augsburg.de/get/llDtPTBNQ1/kenlm_azwtd.scorer.gz) \
Checkpoints TVSMC training with 0.19 WER: [Link](https://megastore.uni-augsburg.de/get/lseBk3Xk9v/tvsmc_0190_ckpt.tar.gz) \
Graph model for above checkpoint: [Link](https://megastore.uni-augsburg.de/get/QmFBEI5s4K/tvsmc_graphs.tar.gz)
