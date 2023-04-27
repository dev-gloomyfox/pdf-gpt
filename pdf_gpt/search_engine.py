from abc import ABC, abstractmethod

import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder  # type: ignore


class SearchEngine(ABC):  # pylint: disable=too-few-public-methods
    """
    A search engine that finds documents relevant to a query in a split document.
    """

    @abstractmethod
    def search(self, query: str, corpus: list[str], k: int) -> list[tuple[int, float]]:
        """
        Query to find the most relevant parts of the split document.
        :param query: Questions to ask
        :param corpus: The contents of the split document.
        :param k: A number for how many related document fragments to fetch.
        :return: A list of tuples with the index and search score of the document fragment.
        The number of lists is related to the value of k.
        """


class CrossEncoderSearchEngine(SearchEngine):  # pylint: disable=too-few-public-methods
    """
    A search engine implemented using a cross-encoder.
    While cross-encoders perform well,
    they suffer from the problem of comparing every sentence pair in a large set of document fragments.
    """

    def __init__(self, cross_encoder: CrossEncoder):
        """
        Cross-encoder search engine constructor.
        :param cross_encoder: The cross encoder object for the sentence transformer.
        For more information, see the following links:
        https://www.sbert.net/examples/applications/cross-encoder/README.html
        """
        self.cross_encoder = cross_encoder

    def search(self, query: str, corpus: list[str], k: int) -> list[tuple[int, float]]:
        sentence_combinations = [[query, corpus_sentence] for corpus_sentence in corpus]
        similarity_scores = self.cross_encoder.predict(sentence_combinations)[:, 1]
        sim_scores_argsort = reversed(np.argsort(similarity_scores))  # type: ignore

        top_k = min(k, len(corpus))

        results = []
        for idx in sim_scores_argsort:
            results.append((idx, similarity_scores[idx]))

        return results[0:top_k]


# mypy: ignore-errors
class SentenceTransformerSearchEngine(
    SearchEngine
):  # pylint: disable=too-few-public-methods
    """
    TODO: Not implements yet
    """

    def __init__(self):
        pass

    def search(self, query: str, corpus: list[str], k: int) -> tuple[int, float]:
        pass
