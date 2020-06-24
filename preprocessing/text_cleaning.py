# -*- coding: utf-8 -*-

import re
import string

import num2words

# ==================================================================================================

# Regex patterns
int_pattern = re.compile(r"[+-]?[0-9]+")
ordinal_pattern = re.compile(r"([0-9]+[.])(?![.0-9])")
float_pattern = re.compile(r"(?<![.0-9])([+-]?[0-9]+[,.][0-9]+)(?![.0-9])")
multi_space_pattern = re.compile(r"\s+")

# Allowed characters a-zA-Z äöüß
allowed = list(string.ascii_lowercase)
allowed.append(" ")
allowed.append("ä")
allowed.append("ö")
allowed.append("ü")
allowed.append("ß")

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
    "-­/:": " ",
}

# Switch keys and value
replacements = {}
for all, replacement in replacer.items():
    for to_replace in all:
        replacements[to_replace] = replacement

# Various replacement rules
special_replacers = {
    " m / s ": "meter pro sekunde",
    "m/s ": "meter pro sekunde",
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
    "‰": "promille",
    "±": "plus minus",
    "°c": "grad celsius",
    "°": "grad",
    "kg": "kilogramm",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
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


def remove_symbols(word):
    """ Remove all symbols that are not allowed. """

    result = word
    bad_characters = []

    for c in result:
        if c not in allowed:
            bad_characters.append(c)

    for c in bad_characters:
        result = result.replace(c, "")

    return result


# ==================================================================================================


def word_to_num(word):
    """ Replace numbers with their written representation. """

    matches = float_pattern.findall(word)
    if len(matches) == 1:
        num = matches[0].replace(",", ".")
        num_word = num2words.num2words(float(num), lang="de")
        word = word.replace(matches[0], " {} ".format(num_word))

    matches = ordinal_pattern.findall(word)
    if len(matches) == 1:
        num_word = num2words.num2words(int(matches[0][:-1]), lang="de", to="ordinal")
        word = word.replace(matches[0], " {} ".format(num_word))
        print(word, num_word)

    matches = int_pattern.findall(word)
    for match in matches:
        num_word = num2words.num2words(int(match), lang="de")
        word = word.replace(match, " {} ".format(num_word))

    # Replace dots like in 168.192.0.1 and make word lowercase again
    word = word.replace(".", " punkt ")
    word = word.lower()
    return word


# ==================================================================================================


def get_bad_character(text):
    """ Return all characters in the text that are not allowed. """

    bad_characters = set()

    for c in text:
        if c not in allowed:
            bad_characters.add(c)

    return bad_characters


# ==================================================================================================


def clean_word(word):
    """
    Clean the given word.

    1. numbers to words
    2. character/rule replacements
    3. delete disallowed symbols
    """

    word = word.lower()
    word = word_to_num(word)
    word = replace_symbols(word)
    word = remove_symbols(word)

    bad_chars = get_bad_character(word)

    if len(bad_chars) > 0:
        print('Bad characters in "{}"'.format(word))
        print("--> {}".format(", ".join(bad_chars)))

    return word


# ==================================================================================================


def clean_sentence(sentence):
    """
    Clean the given sentence.

    1. split into words by spaces
    2. numbers to words
    3. character/rule replacements
    4. delete disallowed symbols
    4. join with spaces
    """

    sentence = re.sub(multi_space_pattern, " ", sentence)
    sentence = sentence.lower()
    sentence = replace_specials(sentence)
    words = sentence.strip().split()

    cleaned_words = []
    for word in words:
        cleaned_word = clean_word(word)
        cleaned_words.append(cleaned_word)

    sentence = " ".join(cleaned_words)
    # Remove duplicate whitespaces again, we may have added some with the above steps
    sentence = re.sub(multi_space_pattern, " ", sentence)
    return sentence
