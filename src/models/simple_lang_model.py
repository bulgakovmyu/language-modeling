from typing import Iterable, Union, Tuple, List
from functools import wraps
import random
import math
import numpy as np
import copy
from src.preprocess.text import TextFileProcessor, make_ngrams_with_freqs, make_ngrams


class BaseLM:
    def _check_fit(func):
        """
        A helper decorator that ensures that the LM was fit on vocab.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.is_fitted:
                raise AttributeError(f"Fit model before call {func.__name__} method")
            return func(self, *args, **kwargs)

        return wrapper

    def __init__(self, n: int, vocab: Iterable[str] = None, unk_label: str = "<UNK>"):
        """
        Language model constructor
        n -- n-gram size
        vocab -- optional fixed vocabulary for the model
        unk_label -- special token that stands in for so-called "unknown" items
        """
        self.n = n
        self._vocab = vocab if vocab else None
        self.unk_label = unk_label
        self.n_grams = None
        self.n_minus_1_grams = None

    def _lookup(self, words: Union[str, Iterable[str]]) -> Union[str, Tuple[str]]:
        """
        Look ups words in the vocabulary
        """
        if isinstance(words, str):
            res = [
                [word if word in self._vocab else self.unk_label for word in sent]
                for sent in TextFileProcessor.make_sentences_from_text(words)
            ]
        elif isinstance(words, list):
            sentences = []
            for sent in words:
                sentences.extend(TextFileProcessor.make_sentences_from_text(sent))
            res = [
                [word if word in self._vocab else self.unk_label for word in sent]
                for sent in sentences
            ]

        return res

    @_check_fit
    def prob(self, word: str, context: Tuple[str] = None) -> float:
        """This method returns probability of a word with given context: P(w_t | w_{t - 1}...w_{t - n + 1})

        For example:
        >>> lm.prob('hello', context=('world',))
        0.99988
        """
        l1 = len(context)
        n_gram = tuple(list(context) + [word])

        try:
            prob = self.n_grams.get(n_gram, 0) / self.n_minus_1_grams.get(context, 0)
        except ZeroDivisionError:
            prob = 0

        return prob

    @_check_fit
    def prob_with_smoothing(
        self, word: str, context: Tuple[str] = None, alpha: float = 1.0
    ) -> float:
        """Proabaility with Additive smoothing

        see: https://en.wikipedia.org/wiki/Additive_smoothing
        where:
        x - count of word in context
        N - total
        d - wocab size
        a - alpha

        """
        l1 = len(context)
        n_gram = tuple(list(context) + [word])

        try:
            prob = (self.n_grams.get(n_gram, 0) + alpha) / (
                self.n_minus_1_grams.get(context, 0) + alpha * len(self._vocab)
            )
        except ZeroDivisionError:
            prob = 0

        return prob

    @_check_fit
    def generate(
        self,
        text_length: int,
        text_seed: Iterable[str] = None,
        random_seed: Union[int, random.Random] = None,
        smoothing=0,
    ) -> List[str]:
        """
        This method generates text of a given length.

        text_length: int -- Length for the output text including `text_seed`.
        text_seed: List[str] -- Given text to calculates probas for next words.
        prob_method: str -- Specifies what method to use: with or without smoothing.

        For example
        >>> lm.generate(2)
        ["hello", "world"]

        """
        if random_seed:
            random.seed(random_seed)
        if text_seed is None:
            generated = []

            ngrams_for_pred = [i[0] for i in self.n_grams.items()]
            counts_for_pred = [i[1] for i in self.n_grams.items()]

            current_word = random.choices(
                ngrams_for_pred,
                list(np.array(counts_for_pred) / np.array(counts_for_pred).sum()),
                k=1,
            )[0][0]
            generated.append(current_word)
        else:
            generated = copy.deepcopy(text_seed)

        for c in range(1, text_length - len(text_seed) + 1):
            if len(generated) < self.n - 1:
                ngrams_for_pred = [
                    i[0]
                    for i in self.n_grams.items()
                    if i[0][len(generated) - 1] == generated[-1]
                ]
                counts_for_pred = [
                    i[1]
                    for i in self.n_grams.items()
                    if i[0][len(generated) - 1] == generated[-1]
                ]

                current_word = random.choices(
                    ngrams_for_pred,
                    list(np.array(counts_for_pred) / np.array(counts_for_pred).sum()),
                    k=1,
                )[0][len(generated)]
            else:
                current_word = random.choices(
                    self._vocab,
                    [
                        self.prob_with_smoothing(
                            word,
                            context=tuple([w for w in generated[-(self.n) + 1 :]]),
                            alpha=smoothing,
                        )
                        if smoothing > 0
                        else self.prob(
                            word, context=tuple([w for w in generated[-(self.n) + 1 :]])
                        )
                        for word in self._vocab
                    ],
                    k=1,
                )[0]
            generated.append(current_word)

        return generated, self.perplexity(generated, smoothing == smoothing)

    def fit(self, sequence_of_tokens: Iterable[str]):
        """
        This method learns probabilities based on given sequence of tokens and
        updates `self.vocab`.

        sequence_of_tokens -- iterable of tokens

        For example
        >>> lm.update(['hello', 'world'])
        """
        if self._vocab is None:
            self._vocab = list(set(sequence_of_tokens))
            self.n_grams = make_ngrams_with_freqs(sequence_of_tokens, n=self.n)
            if self.n >= 2:
                self.n_minus_1_grams = make_ngrams_with_freqs(
                    sequence_of_tokens, n=self.n - 1
                )

        self.is_fitted = True

    @_check_fit
    def perplexity(
        self,
        sequence_of_tokens: Union[Iterable[str], Iterable[Tuple[str]]],
        smoothing: float,
    ) -> float:
        """
        This method returns perplexity for a given sequence of tokens

        sequence_of_tokens -- iterable of tokens
        """
        sequence_ngrams = make_ngrams(sequence_of_tokens, n=self.n)

        if smoothing > 0:
            sequence_probs = [
                self.prob_with_smoothing(
                    ngram[-1],
                    context=tuple([w for w in ngram[:-1]]),
                    alpha=smoothing,
                )
                for ngram in sequence_ngrams
            ]
        else:
            sequence_probs = [
                self.prob(ngram[-1], context=tuple([w for w in ngram[:-1]]))
                for ngram in sequence_ngrams
            ]

        return math.exp(
            (-1 / len(sequence_of_tokens)) * sum(map(math.log, sequence_probs))
        )
