"""
1. Load all corpora where a path is given.
2. Clean transcriptions.
3. Merge all corpora
4. Create Train/Dev/Test splits
5. Export for DeepSpeech
"""

import argparse

import audiomate
import tqdm
from audiomate.corpus import io, subset

import text_cleaning

# ==================================================================================================


def clean_transcriptions(corpus):
    print("Cleaning transcriptions ...")
    for utterance in tqdm.tqdm(corpus.utterances.values()):
        ll = utterance.label_lists[audiomate.corpus.LL_WORD_TRANSCRIPT]

        for label in ll:
            label.value = text_cleaning.clean_sentence(label.value)[0]


# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(description="Prepare data for training.")
    parser.add_argument("target_path", type=str)
    parser.add_argument("--common_voice", type=str)
    parser.add_argument("--css_ten", type=str)
    parser.add_argument("--lingualibre", type=str)
    parser.add_argument("--mailabs", type=str)
    parser.add_argument("--swc", type=str)
    parser.add_argument("--tatoeba", type=str)
    parser.add_argument("--tuda", type=str)
    parser.add_argument("--voxforge", type=str)
    parser.add_argument("--zamia_speech", type=str)
    args = parser.parse_args()

    corpora = []

    if args.common_voice is not None:
        print("Loading common-voice ...")
        corpus = audiomate.Corpus.load(args.common_voice, reader="common-voice")
        corpora.append(corpus)

    if args.css_ten is not None:
        print("Loading css-ten ...")
        corpus = audiomate.Corpus.load(args.css_ten, reader="css10")
        corpora.append(corpus)

    if args.lingualibre is not None:
        print("Loading lingualibre ...")
        corpus = audiomate.Corpus.load(args.lingualibre, reader="lingualibre")
        corpora.append(corpus)

    if args.mailabs is not None:
        print("Loading mailabs ...")
        corpus = audiomate.Corpus.load(args.mailabs, reader="mailabs")
        corpora.append(corpus)

    if args.tatoeba is not None:
        print("Loading tatoeba ...")
        corpus = audiomate.Corpus.load(args.tatoeba, reader="tatoeba")
        corpora.append(corpus)

    if args.swc is not None:
        print("Loading swc ...")
        corpus = audiomate.Corpus.load(args.swc, reader="swc")
        corpora.append(corpus)

    if args.tuda is not None:
        print("Loading tuda ...")
        corpus = audiomate.Corpus.load(args.tuda, reader="tuda")
        corpora.append(corpus)

    if args.voxforge is not None:
        print("Loading voxforge ...")
        corpus = audiomate.Corpus.load(args.voxforge, reader="voxforge")
        corpora.append(corpus)

    if args.zamia_speech is not None:
        print("Loading zamia-speech ...")
        corpus = audiomate.Corpus.load(args.zamia_speech, reader="zamia-speech")
        corpora.append(corpus)

    if len(corpora) <= 0:
        raise ValueError("No Corpus given!")

    merged_corpus = audiomate.Corpus.merge_corpora(corpora)
    clean_transcriptions(merged_corpus)

    print("Splitting corpus ...")
    splitter = subset.Splitter(merged_corpus, random_seed=38)
    split_sizes = {"train": 0.7, "dev": 0.15, "test": 0.15}

    splits = splitter.split(split_sizes, separate_issuers=True)
    if "test" not in splits or "dev" not in splits:
        print("Very small dataset -> using random split instead of separated speakers")
        splits = splitter.split(split_sizes, separate_issuers=False)

    merged_corpus.import_subview("train", splits["train"])
    merged_corpus.import_subview("dev", splits["dev"])
    merged_corpus.import_subview("test", splits["test"])

    print("Saving corpus ...")
    deepspeech_writer = io.MozillaDeepSpeechWriter()
    deepspeech_writer.save(merged_corpus, args.target_path)


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
