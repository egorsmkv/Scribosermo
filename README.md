# DeepSpeech German

_This project is build upon the paper [German End-to-end Speech Recognition based on DeepSpeech](https://www.researchgate.net/publication/336532830_German_End-to-end_Speech_Recognition_based_on_DeepSpeech)._
_Original paper code can be found [here](https://github.com/AASHISHAG/deepspeech-german)._

This project aims to develop a working Speech to Text module using [Mozilla DeepSpeech](https://github.com/mozilla/DeepSpeech), which can be used for any Audio processing pipeline. [Mozillla DeepSpeech](https://github.com/mozilla/DeepSpeech) is a state-of-the-art open-source automatic speech recognition (ASR) toolkit. DeepSpeech is using a model trained by machine learning techniques based on [Baidu's Deep Speech](https://gigaom2.files.wordpress.com/2014/12/deep_speech3_12_17.pdf) research paper. Project DeepSpeech uses Google's TensorFlow to make the implementation easier.

<p align="center">
    <img src="media/deep-speech-v3-1.png" align="center" title="DeepSpeech Graph" />
</p>

## Usage

#### General infos

File structure will look as follows:

```
my_deepspeech_folder
    data_original
    data_prepared
    DeepSpeech
    deepspeech-german    <- This repositiory
```

Clone DeepSpeech and build docker container:

```
git clone https://github.com/mozilla/DeepSpeech.git
docker build -t mozilla_deep_speech DeepSpeech/
```

Build, run and connect to our docker container:
```
cd deepspeech-german
docker build -t deep_speech_german .

docker-compose -f docker-compose.yml up
docker exec -it deepspeech-german_deep_speech_1 bash    # In a new shell
```

#### Download and prepare voice data

* [German Distant Speech Corpus (TUDA-De)](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/acoustic-models.html) ~108h
* [Mozilla Common Voice](https://voice.mozilla.org/) ~325h
* [Voxforge](http://www.voxforge.org/home/forums/other-languages/german/open-speech-data-corpus-for-german) ~35h
* [Spoken Wikipedia Corpora (SWC)](https://nats.gitlab.io/swc/) ~386h
* [M-AILABS Speech Dataset](https://www.caito.de/2019/01/the-m-ailabs-speech-dataset/) ~237h


* [Forschergeist](https://forschergeist.de/archiv/) ~100-150h, but no data pipline existing
<br/>

Download datasets (Run in docker container):
```
python3 deepspeech-german/data/download_data.py --tuda data_original/
python3 deepspeech-german/data/download_data.py --voxforge data_original/
python3 deepspeech-german/data/download_data.py --mailabs data_original/
python3 deepspeech-german/data/download_data.py --swc data_original/
```

Download common voice dataset: https://voice.mozilla.org/en/datasets \
Extract and move it to datasets directory (data_original/common_voice/)

<br/>

Prepare datasets, this may take some minutes (Run in docker container):
```
# Prepare the datasets one by one first to ensure everything is working:

deepspeech-german/pre-processing/run_to_utf_8.sh "../../data_original/voxforge/*/etc/prompts-original"
python3 deepspeech-german/pre-processing/prepare_data.py --voxforge data_original/voxforge/  data_prepared/voxforge/

python3 deepspeech-german/pre-processing/prepare_data.py --tuda data_original/tuda/  data_prepared/tuda/
python3 deepspeech-german/pre-processing/prepare_data.py --common_voice data_original/common_voice/  data_prepared/common_voice/
python3 deepspeech-german/pre-processing/prepare_data.py --mailabs data_original/mailabs/  data_prepared/mailabs/
python3 deepspeech-german/pre-processing/prepare_data.py --swc data_original/swc/  data_prepared/swc/


# To combine multiple datasets run the command as follows:
python3 deepspeech-german/pre-processing/prepare_data.py --tuda data_original/tuda/ --voxforge data_original/voxforge/  data_prepared/tuda_voxforge/

# Or, which is much faster, but only combining train, dev, test and all csv files, run:
python3 deepspeech-german/data/combine_datasets.py data_prepared/ --tuda --voxforge


# To shuffle and replace "äöü" characters and clean the files run (for all 3 csv files):
python3 deepspeech-german/data/dataset_operations.py data_prepared/tuda-voxforge/train.csv data_prepared/tuda-voxforge/train_azc.csv --replace --shuffle --clean

# To split tuda into the correct train, dev and test splits run: 
# (you will have to rename the [train/dev/test]_s.csv files before combining them with other datasets)
python3 deepspeech-german/data/split_tuda.py data_prepared/tuda/ _s
```

Preparation times using Intel i7-8700K:
* voxforge: some seconds
* tuda: some minutes
* mailabs: ~20min
* common_voice: ~3h
* swc: ~6h

#### Create the language model

Download and prepare the open-source [German Speech Corpus](http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/German_sentences_8mil_filtered_maryfied.txt.gz):
```
cd data_original
wget http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/German_sentences_8mil_filtered_maryfied.txt.gz
gzip -d German_sentences_8mil_filtered_maryfied.txt.gz


cd my_deepspeech_folder
python3 deepspeech-german/pre-processing/prepare_vocab.py data_original/German_sentences_8mil_filtered_maryfied.txt data_prepared/clean_vocab.txt
```

Build the Language Model with [KenLM](https://github.com/kpu/kenlm.git) (Run in docker container):
```
native_client/kenlm/build/bin/lmplz --text data_prepared/clean_vocab.txt --arpa data_prepared/words.arpa --o 3    # Add "-S 50%" to only use 50% of memory
native_client/kenlm/build/bin/build_binary -T -s data_prepared/words.arpa data_prepared/lm.binary
```

Build Trie from the generated language model  (Run in docker container):
```
native_client/generate_trie deepspeech-german/data/alphabet.txt data_prepared/lm.binary data_prepared/trie
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

#### Training

Download pretrained deepspeech checkpoints.

```
cd data_original
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.6.0/deepspeech-0.6.0-checkpoint.tar.gz
tar xvfz deepspeech-0.6.0-checkpoint.tar.gz
# Don't delete the compressed file, you may need it later for a new training as the uncompressed will be overwritten
```

Adjust the parameters to your needs (Run in docker container):

```
# Delete old model files:
rm -rf /root/.local/share/deepspeech/summaries && rm -rf /root/.local/share/deepspeech/checkpoints


# Run training
python3 DeepSpeech.py --train_files data_prepared/voxforge/train.csv --dev_files data_prepared/voxforge/dev.csv --test_files data_prepared/voxforge/test.csv \
--alphabet_config_path deepspeech-german/data/alphabet.txt --lm_trie_path data_prepared/trie --lm_binary_path data_prepared/lm.binary --test_batch_size 48 --train_batch_size 48 --dev_batch_size 48 \
--epochs 75 --learning_rate 0.0005 --dropout_rate 0.40 --export_dir deepspeech-german/models --use_allow_growth --use_cudnn_rnn 

python3 DeepSpeech.py --train_files data_prepared/voxforge/train_azc.csv --dev_files data_prepared/voxforge/dev_azc.csv --test_files data_prepared/voxforge/test_az.csv \
--alphabet_config_path deepspeech-german/data/alphabet_az.txt --lm_trie_path data_prepared/trie_az4 --lm_binary_path data_prepared/lm_az4.binary --test_batch_size 48 --train_batch_size 48 --dev_batch_size 48 \
--epochs 75 --learning_rate 0.0005 --dropout_rate 0.40 --export_dir deepspeech-german/models/voxforge/ --use_allow_growth --use_cudnn_rnn --checkpoint_dir data_original/deepspeech-0.6.0-checkpoint/ \
--augmentation_freq_and_time_masking --augmentation_pitch_and_tempo_scaling --augmentation_spec_dropout_keeprate 0.9 --augmentation_speed_up_std 0.2

python3 DeepSpeech.py --train_files data_prepared/tuda-voxforge-swc-mailabs-common_voice/train_az.csv --dev_files data_prepared/tuda-voxforge-swc-mailabs-common_voice/dev_az.csv --test_files data_prepared/tuda-voxforge-swc-mailabs-common_voice/test_az.csv \
--alphabet_config_path deepspeech-german/data/alphabet_az.txt --lm_trie_path data_prepared/trie_az --lm_binary_path data_prepared/lm_az.binary --test_batch_size 12 --train_batch_size 12 --dev_batch_size 12 \
--epochs 75 --learning_rate 0.0005 --dropout_rate 0.40 --export_dir deepspeech-german/models/tvsmc/ --use_allow_growth  --use_cudnn_rnn --checkpoint_dir data_original/deepspeech-0.6.0-checkpoint/ \
--augmentation_freq_and_time_masking --augmentation_pitch_and_tempo_scaling --augmentation_spec_dropout_keeprate 0.9 --augmentation_speed_up_std 0.2


# Run test only
python3 DeepSpeech.py --test_files data_prepared/voxforge/test_az.csv \
--alphabet_config_path deepspeech-german/data/alphabet_az.txt --lm_trie_path data_prepared/trie_az --lm_binary_path data_prepared/lm_az.binary --test_batch_size 48
```

Training time for voxforge on 2x Nvidia 1080Ti using batch size of 48 is about 01:45min per epoch. Training until early stop took 22min for 10 epochs. 

One epoch in tuda with batch size of 12 on single gpu needs about 1:15h. With both gpus it takes about 26min. 

One epoch in mailabs with batch size of 24/12/12 needs about 19min, testing about 21 min. 

One epoch in common_voice with batch size of 24/12/12 needs about 31min, testing needs the same time. Training with augmentation until early stop took 4:20h for 8 epochs. 

One epoch in swc with batch size of 12/12/12 needs about 1:08h, testing about 17 min.

## Results

Some results from the findings in the paper [German End-to-end Speech Recognition based on DeepSpeech](https://www.researchgate.net/publication/336532830_German_End-to-end_Speech_Recognition_based_on_DeepSpeech):

- Mozilla 79.7%
- Voxforge 72.1%
- Tuda-De 26.8%
- Tuda-De+Mozilla 57.3%
- Tuda-De+Voxforge 15.1%
- Tuda-De+Voxforge+Mozilla 21.5%

<br/>

Some results with the current code version: 

| Dataset | Additional Infos | Result |
|---------|------------------|--------|
| Voxforge | | WER: 0.676611, CER: 0.403916, loss: 82.185226 |
| Voxforge | with augmentation | WER: 0.671032, CER: 0.394428, loss: 84.415947 |
| Voxforge | without "äöü" | WER: 0.646702, CER: 0.364471, loss: 82.567413 |
| Voxforge | 5-gram language model, without "äöü" | WER: 0.665490, CER: 0.394863, loss: 82.016052 |
| Voxforge | 4-gram language model, without "äöü" | WER: 0.642716, CER: 0.376940, loss: 80.501076 |
| Voxforge | cleaned data, without "äöü" | WER: 0.634828, CER: 0.353037, loss: 81.905258 |
| Voxforge | above checkpoint, tested on not cleaned data | WER: 0.634556, CER: 0.352879, loss: 81.849220 |
| Voxforge | checkpoint from english deepspeech, without "äöü" | WER: 0.394064, CER: 0.190184, loss: 49.066357 |
| Voxforge | checkpoint from english deepspeech, with augmentation, 4-gram language model, cleaned train and dev data, without "äöü" | WER: 0.434943, CER: 0.214896, loss: 52.146347 |
| Tuda | without "äöü", cleaned train and dev data | WER: 0.412830, CER: 0.236580, loss: 121.374710 |
| Tuda | without "äöü", cleaned train and dev data, dropout 0.25, learning rate 0.0001 | WER: 0.436935, CER: 0.230252, loss: 132.031647 |
| Tuda | correct train/dev/test splitting, without "äöü", cleaned train and dev data, dropout 0.25, learning rate 0.0001 | WER: 0.683900, CER: 0.394106, loss: 167.296478 |
| CommonVoice | with augmentation, dataset with 325h validated | WER: 0.487191, CER: 0.251260, loss: 44.828957 |
| Tuda + Voxforge | without "äöü", checkpoint from english deepspeech, cleaned train and dev data | WER: 0.740130, CER: 0.462036, loss: 156.115921 |
| Tuda + Voxforge | first tuda then voxforge, without "äöü", cleaned train and dev data, dropout 0.25, learning rate 0.0001 | WER: 0.653841, CER: 0.384577, loss: 159.509476 |

#### Trained Language Model, Trie, Speech Model and Checkpoints
TODO
