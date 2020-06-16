"""
1. Load all corpora where a path is given.
2. Clean transcriptions.
3. Merge all corpora
4. Create Train/Dev/Test splits
5. Export for DeepSpeech
"""

import argparse

import audiomate
import text_cleaning
import tqdm
from audiomate.corpus import io
from audiomate.corpus import subset


# ==================================================================================================


def clean_transcriptions(corpus):
    print("Cleaning transcriptions ...")
    for utterance in tqdm.tqdm(corpus.utterances.values()):
        ll = utterance.label_lists[audiomate.corpus.LL_WORD_TRANSCRIPT]

        for label in ll:
            label.value = text_cleaning.clean_sentence(label.value)


# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(description="Prepare data for training.")
    parser.add_argument("target_path", type=str)
    parser.add_argument("--tuda", type=str)
    parser.add_argument("--voxforge", type=str)
    parser.add_argument("--swc", type=str)
    parser.add_argument("--mailabs", type=str)
    parser.add_argument("--common_voice", type=str)
    parser.add_argument("--tatoeba", type=str)

    args = parser.parse_args()

    tuda_path = args.tuda
    voxforge_path = args.voxforge
    swc_path = args.swc
    mailabs_path = args.mailabs
    cv_path = args.common_voice
    tatoeba_path = args.tatoeba

    corpora = []

    if tuda_path is not None:
        print("Loading tuda ...")
        corpus = audiomate.Corpus.load(tuda_path, reader="tuda")
        corpora.append(corpus)

    if voxforge_path is not None:
        print("Loading voxforge ...")
        corpus = audiomate.Corpus.load(voxforge_path, reader="voxforge")
        corpora.append(corpus)

    if swc_path is not None:
        print("Loading swc ...")
        corpus = audiomate.Corpus.load(swc_path, reader="swc")
        corpora.append(corpus)

    if mailabs_path is not None:
        print("Loading mailabs ...")
        corpus = audiomate.Corpus.load(mailabs_path, reader="mailabs")
        corpora.append(corpus)

    if cv_path is not None:
        print("Loading common-voice ...")
        corpus = audiomate.Corpus.load(cv_path, reader="common-voice")
        corpora.append(corpus)

    if tatoeba_path is not None:
        print("Loading tatoeba ...")
        corpus = audiomate.Corpus.load(tatoeba_path, reader="tatoeba")
        corpora.append(corpus)

    if len(corpora) <= 0:
        raise ValueError("No Corpus given!")

    merged_corpus = audiomate.Corpus.merge_corpora(corpora)
    clean_transcriptions(merged_corpus)

    print("Splitting corpus ...")
    splitter = subset.Splitter(merged_corpus, random_seed=38)
    splits = splitter.split(
        {"train": 0.7, "dev": 0.15, "test": 0.15}, separate_issuers=True
    )

    merged_corpus.import_subview("train", splits["train"])
    merged_corpus.import_subview("dev", splits["dev"])
    merged_corpus.import_subview("test", splits["test"])

    print("Saving corpus ...")
    deepspeech_writer = io.MozillaDeepSpeechWriter()
    deepspeech_writer.save(merged_corpus, args.target_path)


# ==================================================================================================

if __name__ == "__main__":
    main()
