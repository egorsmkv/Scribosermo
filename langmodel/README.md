# Building the language model

Instructions to collect sample sentences and build an additional language model for improved accuracy.

<br/>

Download sentence collection datasets from [tuda](http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/)
and [europarl+news](https://www.statmt.org/wmt13/translation-task.html):

```bash
export LANGUAGE="de"
mkdir /data_original/texts/ && mkdir /data_original/texts/${LANGUAGE}/
mkdir /data_prepared/texts/ && mkdir /data_prepared/texts/${LANGUAGE}/

# German
cd /data_original/texts/${LANGUAGE}/
wget "http://ltdata1.informatik.uni-hamburg.de/kaldi_tuda_de/German_sentences_8mil_filtered_maryfied.txt.gz" -O tuda_sentences.txt.gz
gzip -d tuda_sentences.txt.gz

# German or French or Spanish
cd /data_original/texts/
wget "https://www.statmt.org/wmt13/training-monolingual-nc-v8.tgz" -O news-commentary.tgz
tar zxvf news-commentary.tgz && mv training/news-commentary-v8.${LANGUAGE} ${LANGUAGE}/news-commentary-v8.txt
rm news-commentary.tgz && rm -r training/
wget "https://www.statmt.org/wmt13/training-monolingual-europarl-v7.tgz" -O europarl.tgz
tar zxvf europarl.tgz && mv training/europarl-v7.${LANGUAGE} ${LANGUAGE}/europarl-v7.txt
rm europarl.tgz && rm -r training/
# If you have enough space you can also download the other years (2007-2011)
wget "https://www.statmt.org/wmt13/training-monolingual-news-2012.tgz" -O news-2012.tgz
tar zxvf news-2012.tgz && mv training-monolingual/news.2012.${LANGUAGE}.shuffled ${LANGUAGE}/news.2012.txt
rm news-2012.tgz && rm -r training-monolingual/
```

In addition to above sentence collections, we can extract more sentences from our own datasets: \
(Only use the training partitions for that)

```bash
# Run in container
export LANGUAGE="de"

python3 /Scribosermo/langmodel/extract_transcripts.py \
  --input_csv /data_prepared/${LANGUAGE}/librispeech/train-all.csv \
  --output_txt /data_original/texts/${LANGUAGE}/librispeech.txt
```

<br/>

Prepare the sentences.

```bash
# Run in container
export LANGUAGE="de"

python3 /Scribosermo/langmodel/prepare_vocab.py \
  --input_dir /data_original/texts/${LANGUAGE}/ \
  --output_dir /data_prepared/texts/${LANGUAGE}/

# Combine to single file and shuffle it
mkdir /data_prepared/langmodel/ && mkdir /data_prepared/langmodel/${LANGUAGE}/
echo /data_prepared/texts/${LANGUAGE}/*.txt | xargs cat | shuf > /data_prepared/langmodel/${LANGUAGE}/all.txt
```

<br/>

Create an `arpa`-format language model: \
There are two different tools to choose from, _KenLM_ and _PocoLM_, the first is much faster, while the other has an improved optimization.
So the recommendation is to use _KenLM_ for the first and _PocoLM_ for the final tests.

Building `.arpa` file with KenLM:

```bash
# Run in container
export LANGUAGE="de"
cd /data_prepared/langmodel/${LANGUAGE}/

# Build arpa file
/kenlm/build/bin/lmplz --order 5 --temp_prefix /tmp/ --memory 95% --prune 0 0 1 \
  --text all.txt --arpa kenlm.arpa

# Collect and save top-k words
python3 /Scribosermo/langmodel/collect_topk.py --top_k 500000 \
  --input_file all.txt --output_file vocab.txt

# Filter the language model with our vocabulary
/kenlm/build/bin/filter single \
  model:kenlm.arpa lm.arpa < vocab.txt
```

Building `.arpa` file with PocoLM:

```bash
# Run in container
export LANGUAGE="de"
cd /data_prepared/langmodel/${LANGUAGE}/

# Make sure to delete all files except the all.txt in this directory

# Use 5% as dev partition and the other 95% for training
head -n $(expr $(cat all.txt | wc -l) / 100 \* 5) < all.txt > dev.txt
tail -n +$(expr $(cat all.txt | wc -l) / 100 \* 5) < all.txt > train.txt
rm all.txt

# Build 5-gram LM with 95% of available system memory
# This takes about 2h for a 1GB text file and requires about 60GB temporary disk space
python3 /pocolm/scripts/train_lm.py --num-words=500000 --num-splits=$(nproc) \
  --warm-start-ratio=20 --max-memory=$(expr $(free -g | awk '/^Mem:/{print $7}') \* 95 / 100)G \
  ./ 5 ./pocolm/

# Prune arpa model
# 50M n-grams -> ~250MB scorer size (for faster inference), takes about 2h
# 165M n-grams -> ~950MB scorer size (similar to DeepSpeech's scorer), did take about 6h
python3 /pocolm/scripts/prune_lm_dir.py --target-num-ngrams 50000000 \
  ./pocolm/500000_5.pocolm/ ./pocolm/500000_5_pruned.pocolm/

# Convert from custom format to .arpa file
python3 /pocolm/scripts/format_arpa_lm.py ./pocolm/500000_5_pruned.pocolm/ > lm.arpa

# Clean up the vocabulary file
sed 's/ .*//g' ./pocolm/500000_5_pruned.pocolm/words.txt > vocab.txt
```

<br/>

Optimize the model and convert it to _DeepSpeech's_ `.scorer`-format:

```bash
# Reduce model size
/kenlm/build/bin/build_binary -a 255 -q 8 -v trie \
  /data_prepared/langmodel/${LANGUAGE}/lm.arpa \
  /data_prepared/langmodel/${LANGUAGE}/lm.binary

# Optimized scorer alpha and beta values:
# English (taken from DeepSpeech repo): --default_alpha 0.931289039105002 --default_beta 1.1834137581510284
# German: --default_alpha 0.7842902115058261 --default_beta 0.6346150241906542
# Spanish+French: --default_alpha 0.749166959347089 --default_beta 1.6627453128820517
# Italian: --default_alpha 0.910619981788069 --default_beta 0.15660475671195578
# Polish: --default_alpha 1.3060110864019918 --default_beta 3.5010876706821334

export LANGUAGE="de"
/DeepSpeech/data/lm/generate_scorer_package \
  --alphabet /Scribosermo/data/${LANGUAGE}/alphabet.txt \
  --lm /data_prepared/langmodel/${LANGUAGE}/lm.binary \
  --vocab /data_prepared/langmodel/${LANGUAGE}/vocab.txt \
  --package /data_prepared/langmodel/${LANGUAGE}.scorer \
  --default_alpha 0.7842902115058261 --default_beta 0.6346150241906542
```

Clean up intermediate files:

```bash
rm -rf /data_prepared/langmodel/${LANGUAGE}/
```
