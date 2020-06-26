import collections
import re
import string
from functools import partial
from multiprocessing import Pool

import num2words
import tqdm

# ==================================================================================================

# Regex patterns, see www.regexr.com for good explanation
dp = r"(?:[\s][+-])?[0-9]+(?:(?:[.][0-9]{3}(?:(?=[^0-9])))+)?(?:[,][0-9]+)?"
decimal_pattern = re.compile(dp)
ordinal_pattern = re.compile(r"[0-9]+[.] ")
special_pattern = re.compile(r"&#[0-9]+;")
multi_space_pattern = re.compile(r"\s+")

# Allowed characters a-zA-Z äöüß
allowed = list(string.ascii_lowercase)
allowed.append(" ")
allowed.append("ä")
allowed.append("ö")
allowed.append("ü")
allowed.append("ß")
all_bad_characters = set()

umlaut_replacers = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}
umlaut_pattern = re.compile("|".join(umlaut_replacers.keys()))

# Replacement characters
replacer = {
    "àáâãåāăąǟǡǻȁȃȧ": "a",
    "æǣǽ": "ä",
    "çćĉċč": "c",
    "ďđ": "d",
    "èéêëēĕėęěȅȇȩε": "e",
    "ĝğġģǥǧǵ": "g",
    "ĥħȟ": "h",
    "ìíîïĩīĭįıȉȋ": "i",
    "ĵǰ": "j",
    "ķĸǩǩκ": "k",
    "ĺļľŀł": "l",
    "м": "m",
    "ñńņňŉŋǹ": "n",
    "òóôõøōŏőǫǭǿȍȏðο": "o",
    "œ": "ö",
    "ŕŗřȑȓ": "r",
    "śŝşšș": "s",
    "ţťŧț": "t",
    "ùúûũūŭůűųȕȗ": "u",
    "ŵ": "w",
    "ýÿŷ": "y",
    "źżžȥ": "z",
    '–-­/:(),.!?"[]': " ",
}

# Switch keys and value
replacements = {}
for all, replacement in replacer.items():
    for to_replace in all:
        replacements[to_replace] = replacement

# Various replacement rules
special_replacers = {
    " m / s ": "meter pro sekunde",
    " m/s ": "meter pro sekunde",
    "€": "euro",
    "$": "dollar",
    "£": "pfund",
    "%": "prozent",
    "‰": "promille",
    "&": "und",
    "§": "paragraph",
    "m³": "kubikmeter",
    "km²": "quadratkilometer",
    "m²": "quadratmeter",
    "co2": "c o zwei",
    " ft ": "fuss",
    "±": "plus minus",
    "°c": "grad celsius",
    "°": "grad",
    " kg ": "kilogramm",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "@": "at",
    "½": "einhalb",
    "⅓": "ein drittel",
    "¼": "ein viertel",
    "1910er": "neunzehnhundertzehner",
    "1920er": "neunzehnhundertzwanziger",
    "1930er": "neunzehnhundertdreißiger",
    "1940er": "neunzehnhundertvierziger",
    "1950er": "neunzehnhundertfünfziger",
    "1960er": "neunzehnhundertsechziger",
    "1970er": "neunzehnhundertsiebziger",
    "1980er": "neunzehnhundertachtziger",
    "1990er": "neunzehnhundertneunziger",
    "10er": "zehner",
    "20er": "zwanziger",
    "30er": "dreißiger",
    "40er": "vierziger",
    "50er": "fünfziger",
    "60er": "sechziger",
    "70er": "siebziger",
    "80er": "achtziger",
    "90er": "neunziger",
    "eins punkt null null null punkt null null null punkt null null null": "eine milliarde",
    "punkt null null null punkt null null null punkt null null null": "milliarden",
    "eins punkt null null null punkt null null null": "eine million",
    "punkt null null null punkt null null null": "millionen",
}


# ==================================================================================================


def replace_specials(word):
    """ Apply special replacement rules to the given word. """

    for to_replace, replacement in special_replacers.items():
        word = word.replace(to_replace, " {} ".format(replacement))

    return word


# ==================================================================================================


def replace_symbols(word):
    """ Apply all replacement character rules to the given word. """

    for to_replace, replacement in replacements.items():
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
        num_word = num2words.num2words(int(matches[0][:-1]), lang="de", to="ordinal")
        word = word.replace(matches[0], " {} ".format(num_word))

    matches = decimal_pattern.findall(word)
    for match in matches:
        num = match.replace(".", "")
        num = num.replace(",", ".")
        num_word = num2words.num2words(float(num), lang="de")
        word = word.replace(match, " {} ".format(num_word))

    # Make word lowercase again
    word = word.lower()
    return word


# ==================================================================================================


def get_bad_characters(text):
    """ Return all characters in the text that are not allowed. """

    bad_characters = []

    for c in text:
        if c not in allowed:
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
    # Replace special characters again, sometimes they are behind a number like 12kg
    word = replace_specials(word)
    word = replace_symbols(word)

    bad_chars = get_bad_characters(word)
    word = remove_symbols(word, bad_chars)

    return word, bad_chars


# ==================================================================================================


def clean_sentence(sentence, replace_umlauts=False):
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
    if replace_umlauts:
        sentence = umlaut_pattern.sub(lambda x: umlaut_replacers[x.group()], sentence)
    # Remove duplicate whitespaces again, we may have added some with the above steps
    sentence = re.sub(multi_space_pattern, " ", sentence)

    return sentence, bad_chars_sen


# ==================================================================================================


def clean_sentence_list(sentences, replace_umlauts):
    cl_func = partial(clean_sentence, replace_umlauts=replace_umlauts)

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

    return cleaned_sentences
