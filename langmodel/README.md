# Building the language model

Instructions to collect sample sentences and build an additional language model for improved accuracy.

<br/>

Download sentence collection datasets from [tuda](http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/)
and [europarl+news](https://www.statmt.org/wmt13/translation-task.html):
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
```

In addition to above sentence collections, we can extract more sentences from our own datasets: \
(Only use the training partitions for that)
```bash
# Run in container
export LANGUAGE="de"

python3 /deepspeech-polyglot/langmodel/extract_transcripts.py \
  --input_csv /data_prepared/${LANGUAGE}/librispeech/train.csv \
  --output_txt /data_original/texts/${LANGUAGE}/librispeech.txt 
```

<br/>

Prepare the sentences.

<br/>

Create the language model: \
(If you want to keep the intermediate `.arpa` files, you can add the `--keep_arpa`)
```bash
# Run in container
export LANGUAGE="de"

python3 /deepspeech-polyglot/langmodel/generate_lm.py \
  --input_dir /data_prepared/texts/${LANGUAGE}/ \
  --output_dir /data_prepared/texts/${LANGUAGE}/ \
  --top_k 500000 --kenlm_bins /DeepSpeech/native_client/kenlm/build/bin/ \
  --arpa_order 5 --max_arpa_memory "85%" --arpa_prune "0|0|1" \
  --binary_a_bits 255 --binary_q_bits 8 --binary_type trie --discount_fallback
```

Combine files to the scorer:
```bash
# Run in container
export LANGUAGE="de"

# Get prebuilt generator script
cd /DeepSpeech/data/lm/
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/native_client.amd64.cpu.linux.tar.xz
tar xvf native_client.*.tar.xz

# Optimized scorer alpha and beta values:
# English (taken from DeepSpeech repo): --default_alpha 0.931289039105002 --default_beta 1.1834137581510284
# German: --default_alpha 0.7842902115058261 --default_beta 0.6346150241906542
# Spanish: --default_alpha 0.749166959347089 --default_beta 1.6627453128820517
# French: --default_alpha 0.9000153993017823 --default_beta 2.478779501401466
# Italian: --default_alpha 0.910619981788069 --default_beta 0.15660475671195578
# Polish: --default_alpha 1.3060110864019918 --default_beta 3.5010876706821334

/DeepSpeech/data/lm/generate_scorer_package \
  --alphabet /deepspeech-polyglot/data/alphabet_${LANGUAGE}.txt \
  --lm /data_prepared/texts/${LANGUAGE}/lm.binary \
  --vocab /data_prepared/texts/${LANGUAGE}/vocab-500000.txt \
  --package /data_prepared/texts/${LANGUAGE}/kenlm_${LANGUAGE}.scorer \
  --default_alpha 0.8223176270809696 --default_beta 0.25566134318440037
```
