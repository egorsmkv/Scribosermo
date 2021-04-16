import argparse
from collections import Counter

import tqdm

# ==================================================================================================


def collect_topk(input_file: str, output_file: str, top_k: int):
    """Convert to lowercase, count word occurrences and save top-k words to a file"""

    # Conversion and counting
    print("Counting word occurrences ...")
    counter = Counter()
    with open(input_file, "r", encoding="utf-8") as file:
        for line in tqdm.tqdm(file.readlines()):
            counter.update(line.split())

    # Save top-k words
    print("\nSaving top {} words ...".format(top_k))
    top_counter = counter.most_common(top_k)
    vocab_str = "\n".join(word for word, count in top_counter)
    with open(output_file, "w+") as file:
        file.write(vocab_str)

    # Statistics
    print("\nCalculating word statistics ...")
    total_words = sum(counter.values())
    print("  Your text file has {} words in total".format(total_words))
    print("  It has {} unique words".format(len(counter)))

    top_words_sum = sum(count for word, count in top_counter)
    word_fraction = (top_words_sum / total_words) * 100
    msg = "  Your top-{} words cover {:.4f} percent of all words"
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


# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Collect top-k words for ARPA filtering."
    )
    parser.add_argument(
        "--input_file",
        help="Path to the file.txt with sample sentences",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file", help="Path for the output vocabulary", type=str, required=True
    )
    parser.add_argument(
        "--top_k",
        help="Collect top_k most frequent words which are used to filter the ARPA file.",
        type=int,
        required=True,
    )
    args = parser.parse_args()

    collect_topk(args.input_file, args.output_file, args.top_k)


# ==================================================================================================

if __name__ == "__main__":
    main()
