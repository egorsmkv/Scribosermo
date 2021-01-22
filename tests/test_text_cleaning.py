import os
import sys

# The if block is required for isort
if True:  # pylint: disable=using-constant-test
    file_path = os.path.dirname(os.path.realpath(__file__)) + "/"
    sys.path.insert(1, file_path + "../preprocessing")
    import text_cleaning


# ==================================================================================================


def test_clean_sentence():
    """ Only single sentence needed for output format test and coverage """

    os.environ["LANGUAGE"] = "de"
    text_cleaning.load_language()

    sentences = "Hi #, wie geht's dir?"
    correct_sentence = "hi wie geht's dir"
    correct_deleted = ["#"]
    cleaned_sentence, deleted_chars = text_cleaning.clean_sentence(sentences)

    assert cleaned_sentence == correct_sentence
    assert deleted_chars == correct_deleted


# ==================================================================================================


def test_clean_sentence_list_de():
    os.environ["LANGUAGE"] = "de"
    text_cleaning.load_language()

    sentences = [
        "Hi, wie geht's dir?",
        "Möchtest du 3kg Vanilleeis?",
        "Ich habe leider nur 2€",
        "Der Preiß mag dafür 12.300,50€",
        "Für Vanilleeis? Da kauf ich lieber 1,5m² Grundstück in München",
    ]
    correct_sentences = [
        "hi wie geht's dir",
        "moechtest du drei kilogramm vanilleeis",
        "ich habe leider nur zwei euro",
        "der preiss mag dafuer zwoelftausenddreihundert komma fuenf euro",
        "fuer vanilleeis da kauf ich lieber eins komma fuenf quadratmeter grundstueck in muenchen",
    ]
    cleaned_sentences = text_cleaning.clean_sentence_list(sentences)

    assert cleaned_sentences == correct_sentences


# ==================================================================================================


def test_clean_sentence_list_es():
    os.environ["LANGUAGE"] = "es"
    text_cleaning.load_language()

    sentences = ["¿Quién quiere casarse ...?"]
    correct_sentences = [
        "quién quiere casarse",
    ]
    cleaned_sentences = text_cleaning.clean_sentence_list(sentences)

    assert cleaned_sentences == correct_sentences


# ==================================================================================================


def test_clean_sentence_list_fr():
    os.environ["LANGUAGE"] = "fr"
    text_cleaning.load_language()

    sentences = [
        "«Une chance qu'il est arrivé.",
    ]
    correct_sentences = [
        "une chance qu'il est arrivé",
    ]
    cleaned_sentences = text_cleaning.clean_sentence_list(sentences)

    assert cleaned_sentences == correct_sentences


# ==================================================================================================


def test_clean_sentence_list_it():
    os.environ["LANGUAGE"] = "it"
    text_cleaning.load_language()

    sentences = [
        "sì, niente, insomma: né padre, né madre, né come fu o come non fu;",
    ]
    correct_sentences = [
        "si niente insomma ne padre ne madre ne come fu o come non fu",
    ]
    cleaned_sentences = text_cleaning.clean_sentence_list(sentences)

    assert cleaned_sentences == correct_sentences


# ==================================================================================================


def test_clean_sentence_list_pl():
    os.environ["LANGUAGE"] = "pl"
    text_cleaning.load_language()

    sentences = [
        "Lopez widząc, żem przyszedł do siebie, jął znowu rozwodzić narzekania:",
    ]
    correct_sentences = [
        "lopez widząc żem przyszedł do siebie jął znowu rozwodzić narzekania",
    ]
    cleaned_sentences = text_cleaning.clean_sentence_list(sentences)

    assert cleaned_sentences == correct_sentences
