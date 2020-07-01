import os
import sys

file_path = os.path.dirname(os.path.realpath(__file__)) + "/"
sys.path.insert(1, file_path + "../preprocessing")

import text_cleaning


# ==================================================================================================


def test_clean_sentence():
    """ Only single sentence needed for output format test and coverage """

    lang = "de"
    text_cleaning.load_replacers(lang)

    sentences = "Hi, wie geht's dir?"
    correct_sentence = "hi wie gehts dir"
    correct_deleted = ["'"]
    cleaned_sentence, deleted_chars = text_cleaning.clean_sentence(sentences)

    assert cleaned_sentence == correct_sentence
    assert deleted_chars == correct_deleted


# ==================================================================================================


def test_clean_sentence_list_de():
    lang = "de"
    text_cleaning.load_replacers(lang)

    sentences = [
        "Hi, wie geht's dir?",
        "Möchtest du 3kg Vanilleeis?",
        "Ich habe leider nur 2€",
        "Der Preiß mag dafür 12.300,50€",
        "Für Vanilleeis? Da kauf ich lieber 1,5m² Grundstück in München",
    ]
    correct_sentences = [
        "hi wie gehts dir",
        "moechtest du drei kilogramm vanilleeis",
        "ich habe leider nur zwei euro",
        "der preiss mag dafuer zwoelftausenddreihundert komma fuenf euro",
        "fuer vanilleeis da kauf ich lieber eins komma fuenf quadratmeter grundstueck in muenchen",
    ]
    cleaned_sentences = text_cleaning.clean_sentence_list(sentences)

    assert cleaned_sentences == correct_sentences


# ==================================================================================================


def test_clean_sentence_list_es():
    lang = "es"
    text_cleaning.load_replacers(lang)

    sentences = ["¿Quién quiere casarse ...?"]
    correct_sentences = [
        "quién quiere casarse",
    ]
    cleaned_sentences = text_cleaning.clean_sentence_list(sentences)

    assert cleaned_sentences == correct_sentences


# ==================================================================================================


def test_clean_sentence_list_fr():
    lang = "fr"
    text_cleaning.load_replacers(lang)

    sentences = [
        "«Une chance qu'il est arrivé.",
    ]
    correct_sentences = [
        "une chance qu'il est arrivé",
    ]
    cleaned_sentences = text_cleaning.clean_sentence_list(sentences)

    assert cleaned_sentences == correct_sentences
