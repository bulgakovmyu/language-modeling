from typing import List
import contractions
import re
import string
import itertools
from nltk.lm.preprocessing import pad_both_ends
from nltk.util import ngrams


class TextFileProcessor(object):
    def __init__(self, filepath):
        self.filepath = filepath

    def read_textfile(self):
        with open(self.filepath, "r") as f:
            return f.read()

    def run(self):
        sentences = self.make_sentences_from_text(self.read_textfile())
        whole_text_list, vocab = self.sentences2wholetext(sentences)
        return sentences, whole_text_list, vocab

    @staticmethod
    def sentences2wholetext(sentences: List[List[str]], vocab=None, is_padded=False):
        # finding the vocabulary
        if vocab is None:
            if is_padded == False:
                sentences_with_pads = [pad_both_ends(sent, n=2) for sent in sentences]
                whole_text_list = list(itertools.chain(*sentences_with_pads))
            else:
                whole_text_list = list(itertools.chain(*sentences))
            vocab = list(set(whole_text_list))

        assert isinstance(
            vocab, list
        ), f"The inputing vocabulary needs to be a list, not a {type(vocab)}"

        return whole_text_list, vocab

    @staticmethod
    def make_sentences_from_text(text: str):

        sentences = [
            " ".join([contractions.fix(word) for word in sentence.lower().split(" ")])
            for sentence in re.sub(".*(:\n)", "", text)
            .replace("\n\n", "<EOS><SOS>")
            .replace("\n", " ")
            .replace("--", " ")
            .replace("-", " ")
            .split("<EOS><SOS>")
        ]
        sentences = [
            [w for w in re.findall(r"[\w']+|[.,!?;]", sent) if len(w) > 0]
            for sent in sentences
        ]
        sentences = [
            sent for sent in sentences if not any([1 if "'" in w else 0 for w in sent])
        ]
        return sentences


def make_ngrams_with_freqs(whole_text_list: list, n: int = 2):

    n_grams = list(ngrams(whole_text_list, n=n))

    n_gram_freqs = dict()
    for n_gram in n_grams:
        if n_gram in n_gram_freqs:
            n_gram_freqs[n_gram] += 1
        else:
            n_gram_freqs[n_gram] = 1

    # return sorted(n_gram_freqs.items(), key=lambda gram: -gram[1])
    return n_gram_freqs


def make_ngrams(whole_text_list: list, n: int = 2):

    n_grams = list(ngrams(whole_text_list, n=n))
    return n_grams
