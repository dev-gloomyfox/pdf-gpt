import unittest

from sentence_transformers.cross_encoder import CrossEncoder  # type: ignore

from pdf_gpt.search_engine import CrossEncoderSearchEngine


class SearchEngineTestCase(unittest.TestCase):

    query = "What is cat"
    corpus = [
        "The cat (Felis catus) is a domestic species of small carnivorous mammal.",
        "The dog (Canis familiaris or Canis lupus familiaris) is a domesticated descendant of the wolf.",
        "Cats are commonly kept as house pets but can also be farm cats or feral cats.",
        "Female domestic cats can have kittens from spring to late autumn.",
        "Dogs were the first species to be domesticated by hunter-gatherers over 15,000 years ago.",
    ]

    def test_cross_encoder_search_engine_search(self):
        search_engine = CrossEncoderSearchEngine(
            CrossEncoder(
                "amberoad/bert-multilingual-passage-reranking-msmarco", max_length=512
            )
        )

        results = search_engine.search(self.query, self.corpus, 3)
        idx = []
        for result in results:
            idx.append(result[0])

        self.assertListEqual([0, 2, 1], idx)


if __name__ == "__main__":
    unittest.main()
