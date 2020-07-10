import collections
import json
import os
import re
from functools import partial
from multiprocessing import Pool

import num2words
import tqdm

# ==================================================================================================

file_path = os.path.dirname(os.path.realpath(__file__)) + "/"
lang = None
langdicts = None

# Regex patterns, see www.regexr.com for good explanation
decimal_pattern = None
ordinal_pattern = None
special_pattern = re.compile(r"&#[0-9]+;|&nbsp;")
multi_space_pattern = re.compile(r"\s+")

# Allowed characters
allowed_chars = None
all_bad_characters = set()

# Special replacements
umlaut_replacers = None
special_replacers = None
char_replacers = None


# ==================================================================================================


def load_language():
    """ Load language and replacers in extra function that they are reloadable for testing """

    global lang, langdicts

    lang = os.getenv("LANGUAGE")
    if lang is None:
        print("Cleaning texts with default language 'de'")
        lang = "de"
    else:
        print("Cleaning texts with language '{}'".format(lang))

    if langdicts is None:
        langdicts = load_langdicts()
    load_replacers(lang)


# ==================================================================================================


def load_langdicts() -> dict:
    """ Load the langdicts """

    path = file_path + "../data/langdicts.json"
    with open(path, "r", encoding="utf-8") as file:
        content = json.load(file)
    return content


# ==================================================================================================


def load_replacers(lang):
    global decimal_pattern, ordinal_pattern, allowed_chars, umlaut_replacers, special_replacers, char_replacers

    decimal_pattern = re.compile(langdicts["number_pattern"][lang]["decimal"])
    ordinal_pattern = re.compile(langdicts["number_pattern"][lang]["ordinal"])

    allowed_chars = langdicts["allowed_chars"][lang]
    umlaut_replacers = langdicts["umlaut_replacers"][lang]
    special_replacers = langdicts["special_replacers"][lang]

    replacer = langdicts["char_replacers"][lang]
    char_replacers = {}
    for all, replacement in replacer.items():
        # Switch keys and value
        for to_replace in all:
            char_replacers[to_replace] = replacement


# ==================================================================================================

load_language()


# ==================================================================================================


def replace_specials(word):
    """ Apply special replacement rules to the given word. """

    for to_replace, replacement in special_replacers.items():
        word = word.replace(to_replace, " {} ".format(replacement))

    return word


# ==================================================================================================


def replace_symbols(word):
    """ Apply all replacement character rules to the given word. """

    for to_replace, replacement in umlaut_replacers.items():
        word = word.replace(to_replace, replacement)

    for to_replace, replacement in char_replacers.items():
        word = word.replace(to_replace, replacement)

    return word


# ==================================================================================================


def remove_symbols(word, bad_characters):
    """ Remove all symbols that are not allowed. """

    for c in set(bad_characters):
        word = word.replace(c, "")

    return word


# ==================================================================================================


def word_to_num(word):
    """ Replace numbers with their written representation. """

    matches = special_pattern.findall(word)
    for match in matches:
        word = word.replace(match, " ")

    matches = ordinal_pattern.findall(word)
    if len(matches) == 1:
        num_word = num2words.num2words(int(matches[0][:-1]), lang=lang, to="ordinal")
        word = word.replace(matches[0], " {} ".format(num_word))

    matches = decimal_pattern.findall(word)
    for match in matches:
        num = match.replace(".", "")
        num = num.replace(",", ".")
        num_word = num2words.num2words(float(num), lang=lang)
        word = word.replace(match, " {} ".format(num_word))

    # Make word lowercase again
    word = word.lower()
    return word


# ==================================================================================================


def get_bad_characters(text):
    """ Return all characters in the text that are not allowed. """

    bad_characters = []

    for c in text:
        if c not in allowed_chars:
            bad_characters.append(c)

    return bad_characters


# ==================================================================================================


def clean_word(word):
    """
    Clean the given word.

    1. numbers to words
    2. character replacements
    3. delete disallowed symbols
    """

    word = word.lower()
    word = word_to_num(word)
    # Replace special words again, sometimes they are behind a number like '12kg' -> 'twelve kg'
    # Adding a space because replacer only looks for ' kg ' so that 'kg' is not replaced in words
    word = replace_specials(" {} ".format(word))
    word = replace_symbols(word)

    bad_chars = get_bad_characters(word)
    word = remove_symbols(word, bad_chars)

    return word, bad_chars


# ==================================================================================================


def clean_sentence(sentence):
    """ Clean the given sentence """

    sentence = re.sub(multi_space_pattern, " ", sentence)
    sentence = sentence.lower()
    sentence = replace_specials(sentence)
    words = sentence.strip().split()

    cleaned_words = []
    bad_chars_sen = []
    for word in words:
        cleaned_word, bad_chars = clean_word(word)
        cleaned_words.append(cleaned_word)
        bad_chars_sen.extend(bad_chars)
    sentence = " ".join(cleaned_words)

    # Remove duplicate whitespaces again, we may have added some with the above steps
    sentence = re.sub(multi_space_pattern, " ", sentence)
    sentence = sentence.strip()

    return sentence, bad_chars_sen


# ==================================================================================================


def clean_sentence_list(sentences):
    cl_func = partial(clean_sentence)

    with Pool() as p:
        processed_sentences = list(
            tqdm.tqdm(p.imap(cl_func, sentences), total=len(sentences))
        )

    cleaned_sentences, bad_characters = zip(*processed_sentences)
    all_bad_characters = []

    for bc in bad_characters:
        all_bad_characters.extend(bc)

    msg = "\nCharacters which were deleted without replacement: {}"
    print(msg.format(collections.Counter(all_bad_characters)))

    return list(cleaned_sentences)
