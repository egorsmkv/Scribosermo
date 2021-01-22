import argparse
import gzip
import io
import os
import subprocess
from collections import Counter

import tqdm

# ==================================================================================================


def convert_and_filter_topk(input_dir: str, output_dir: str, top_k: int):
    """Convert to lowercase, count word occurrences and save top-k words to a file"""

    counter = Counter()
    data_lower = os.path.join(output_dir, "lower.txt.gz")

    txt_files = os.listdir(input_dir)
    txt_files = [
        f for f in txt_files if f.endswith(".txt") and not f.startswith("vocab-")
    ]
    txt_files = [os.path.join(input_dir, f) for f in txt_files]
    print("\nFound {} text files".format(len(txt_files)))

    # Conversion and counting
    print("Converting to lowercase and counting word occurrences ...")
    with io.TextIOWrapper(
        io.BufferedWriter(gzip.open(data_lower, "w+")), encoding="utf-8"
    ) as file_out:

        for tfile in txt_files:
            with open(tfile, "r", encoding="utf-8") as source_file:
                lines = source_file.readlines()

            for line in tqdm.tqdm(lines):
                line_lower = line.lower()
                counter.update(line_lower.split())
                file_out.write(line_lower)

    # Save top-k words
    print("\nSaving top {} words ...".format(top_k))
    top_counter = counter.most_common(top_k)
    vocab_str = "\n".join(word for word, count in top_counter)
    vocab_path = "vocab-{}.txt".format(top_k)
    vocab_path = os.path.join(output_dir, vocab_path)
    with open(vocab_path, "w+") as file:
        file.write(vocab_str)

    # Statistics
    print("\nCalculating word statistics ...")
    total_words = sum(counter.values())
    print("  Your text file has {} words in total".format(total_words))
    print("  It has {} unique words".format(len(counter)))

    top_words_sum = sum(count for word, count in top_counter)
    word_fraction = (top_words_sum / total_words) * 100
    msg = "  Your top-{} words are {:.4f} percent of all words"
    print(msg.format(top_k, word_fraction))

    print('  Your most common word "{}" occurred {} times'.format(*top_counter[0]))
    last_word, last_count = top_counter[-1]
    msg = '  The least common word in your top-k is "{}" with {} times'
    print(msg.format(last_word, last_count))

    for i, (w, c) in enumerate(reversed(top_counter)):
        if c > last_count:
            msg = '  The first word with {} occurrences is "{}" at place {}'
            print(msg.format(c, w, len(top_counter) - 1 - i))
            break

    return data_lower, vocab_str


# ==================================================================================================


def build_lm(args, data_lower, vocab_str):
    print("\nCreating ARPA file ...")
    lm_path = os.path.join(args.output_dir, "lm.arpa")
    subargs = [
        os.path.join(args.kenlm_bins, "lmplz"),
        "--order",
        str(args.arpa_order),
        "--temp_prefix",
        args.output_dir,
        "--memory",
        args.max_arpa_memory,
        "--text",
        data_lower,
        "--arpa",
        lm_path,
        "--prune",
        *args.arpa_prune.split("|"),
    ]
    if args.discount_fallback:
        subargs += ["--discount_fallback"]
    subprocess.check_call(subargs)

    # Filter LM using vocabulary of top-k words
    print("\nFiltering ARPA file using vocabulary of top-k words ...")
    filtered_path = os.path.join(args.output_dir, "lm_filtered.arpa")
    subprocess.run(
        [
            os.path.join(args.kenlm_bins, "filter"),
            "single",
            "model:{}".format(lm_path),
            filtered_path,
        ],
        input=vocab_str.encode("utf-8"),
        check=True,
    )

    # Quantize and produce trie binary.
    print("\nBuilding lm.binary ...")
    binary_path = os.path.join(args.output_dir, "lm.binary")
    subprocess.check_call(
        [
            os.path.join(args.kenlm_bins, "build_binary"),
            "-a",
            str(args.binary_a_bits),
            "-q",
            str(args.binary_q_bits),
            "-v",
            args.binary_type,
            filtered_path,
            binary_path,
        ]
    )


# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate lm.binary and top-k vocab for DeepSpeech scorer."
    )
    parser.add_argument(
        "--input_dir",
        help="Path to a directory containing one or multiple file.txt with sample sentences",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir", help="Directory path for the output", type=str, required=True
    )
    parser.add_argument(
        "--top_k",
        help="Use top_k most frequent words for the vocab.txt file. These will be used to filter the ARPA file.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--kenlm_bins",
        help="File path to the KENLM binaries lmplz, filter and build_binary",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--arpa_order",
        help="Order of k-grams in ARPA-file generation",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--max_arpa_memory",
        help="Maximum allowed memory usage for ARPA-file generation",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--arpa_prune",
        help="ARPA pruning parameters. Separate values with '|'",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--binary_a_bits",
        help="Build binary quantization value a in bits",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--binary_q_bits",
        help="Build binary quantization value q in bits",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--binary_type",
        help="Build binary data structure type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--discount_fallback",
        help="To try when such message is returned by kenlm: 'Could not calculate Kneser-Ney discounts [...] rerun with --discount_fallback'",
        action="store_true",
    )
    parser.add_argument(
        "--keep_arpa",
        help="Keep intermediate arpa files instead of deleting them automatically",
        action="store_true",
    )
    args = parser.parse_args()

    data_lower, vocab_str = convert_and_filter_topk(
        args.input_dir, args.output_dir, args.top_k
    )
    build_lm(args, data_lower, vocab_str)

    # Delete intermediate files
    os.remove(os.path.join(args.output_dir, "lower.txt.gz"))
    if not args.keep_arpa:
        os.remove(os.path.join(args.output_dir, "lm.arpa"))
        os.remove(os.path.join(args.output_dir, "lm_filtered.arpa"))


# ==================================================================================================

if __name__ == "__main__":
    main()
