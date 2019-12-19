# Automatic Speech Recognition (ASR) - DeepSpeech German

_This is the project for the paper [German End-to-end Speech Recognition based on DeepSpeech](https://www.researchgate.net/publication/336532830_German_End-to-end_Speech_Recognition_based_on_DeepSpeech) published at [KONVENS 2019](https://2019.konvens.org/)._

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
* [Mozilla Common Voice](https://voice.mozilla.org/) ~340h
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
```


#### Create the language model

We used an open-source [German Speech Corpus](http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/German_sentences_8mil_filtered_maryfied.txt.gz) released by [University of Hamburg](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/acoustic-models.html). \
Download and prepare text:
```
cd data_original
wget http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/German_sentences_8mil_filtered_maryfied.txt.gz
gzip -d German_sentences_8mil_filtered_maryfied.txt.gz


cd my_deepspeech_folder
deepspeech-german/pre-processing/prepare_vocab.py data_original/German_sentences_8mil_filtered_maryfied.txt data_prepared/clean_vocab.txt
```

We used [KenLM](https://github.com/kpu/kenlm.git) toolkit to train a 3-gram language model. It is Language Model inference code by [Kenneth Heafield](https://kheafield.com/). \
Build the Language Model (Run in docker container):
```
kenlm/build/bin/lmplz --text data_prepared/clean_vocab.txt --arpa data_prepared/words.arpa --o 3    # Add "-S 50%" to only use 50% of memory
kenlm/build/bin/build_binary -T -s data_prepared/words.arpa data_prepared/lm.binary
```

Build Trie from the generated language model  (Run in docker container):
```
native_client/generate_trie deepspeech-german/data/alphabet.txt data_prepared/lm.binary data_prepared/trie
```

#### Training

Adjust the parameters to your needs (Run in docker container):

```
# Delete old model files:
rm -rf /root/.local/share/deepspeech/summaries && rm -rf /root/.local/share/deepspeech/checkpoints


# Run training
python3 DeepSpeech.py --train_files data_prepared/voxforge/train.csv --dev_files data_prepared/voxforge/dev.csv --test_files data_prepared/voxforge/test.csv \
--alphabet_config_path deepspeech-german/data/alphabet.txt --lm_trie_path data_prepared/trie --lm_binary_path data_prepared/lm.binary --test_batch_size 48 --train_batch_size 48 --dev_batch_size 48 \
--epochs 75 --learning_rate 0.0005 --dropout_rate 0.40 --export_dir deepspeech-german/models --use_allow_growth --use_cudnn_rnn \
--augmentation_freq_and_time_masking --augmentation_pitch_and_tempo_scaling --augmentation_spec_dropout_keeprate 0.9 --augmentation_speed_up_std 0.2


# Run test only
python3 DeepSpeech.py --test_files data_prepared/voxforge/test.csv \
--alphabet_config_path z_deepspeech/deepspeech-german/data/alphabet.txt --lm_trie_path z_deepspeech/data/trie --lm_binary_path z_deepspeech/data/lm.binary --test_batch_size 48
```

Training time for voxforge on 2x Nvidia 1080Ti using batch size of 48 is about 01:45 min per epoch. Training until early stop took 21:40 min for 10 epochs.

## Results

Some results from our findings in the paper _(Refer our paper for more information)_.

- Mozilla 79.7%
- Voxforge 72.1%
- Tuda-De 26.8%
- Tuda-De+Mozilla 57.3%
- Tuda-De+Voxforge 15.1%
- Tuda-De+Voxforge+Mozilla 21.5%

<br/>

Some results with the current code version:
- Voxforge --- WER: 0.676611, CER: 0.403916, loss: 82.185226
- Voxforge with augmentation --- WER: 0.671032, CER: 0.394428, loss: 84.415947


#### Trained Language Model, Trie, Speech Model and Checkpoints

The DeepSpeech model can be directly re-trained on new dataset. The required dependencies are available at: \
https://drive.google.com/drive/folders/1nG6xii2FP6PPqmcp4KtNVvUADXxEeakk?usp=sharing

DeepSpeech 0.5-Model: https://drive.google.com/file/d/1VN1xPH0JQNKK6DiSVgyQ4STFyDY_rle3/view

## Acknowledgments
* [Prof. Dr.-Ing. Torsten Zesch](https://www.ltl.uni-due.de/team/torsten-zesch) - Co-Author
* [Dipl.-Ling. Andrea Horbach](https://www.ltl.uni-due.de/team/andrea-horbach)
* [Matthias](https://github.com/ynop/audiomate)

 
## References
If you use our findings/scripts in your academic work, please cite:
```
@inproceedings{agarwal-zesch-2019-german,
    author = "Aashish Agarwal and Torsten Zesch",
    title = "German End-to-end Speech Recognition based on DeepSpeech",
    booktitle = "Preliminary proceedings of the 15th Conference on Natural Language Processing (KONVENS 2019): Long Papers",
    year = "2019",
    address = "Erlangen, Germany",
    publisher = "German Society for Computational Linguistics \& Language Technology",
    pages = "111--119"
}
```
<!--  An open-access Arxiv preprint is available here: -->
