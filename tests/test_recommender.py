import unittest
from pathlib import Path

from recommender import Resource, TfidfRecommender, evaluate_hit, load_resources, tokenize

DATA = Path(__file__).parents[1] / "data" / "resources.csv"


class RecommenderTests(unittest.TestCase):
    def test_tokenization(self):
        self.assertEqual(tokenize("Python & Data-Science!"), ["python", "data", "science"])

    def test_loads_resources(self):
        resources = load_resources(DATA)
        self.assertEqual(len(resources), 10)
        self.assertEqual(resources[0].title, "Python Data Analysis with Pandas")

    def test_profile_returns_relevant_topic(self):
        model = TfidfRecommender().fit(load_resources(DATA))
        results = model.recommend("I want to learn tokenization and text classification", 1)
        self.assertEqual(results[0][0].topic, "nlp")
        self.assertGreater(results[0][1], 0)

    def test_level_filter(self):
        model = TfidfRecommender().fit(load_resources(DATA))
        results = model.recommend("neural networks and tensors", 5, level="advanced")
        self.assertTrue(results)
        self.assertTrue(all(resource.level == "advanced" for resource, _ in results))

    def test_evaluation(self):
        model = TfidfRecommender().fit(load_resources(DATA))
        profiles = [("probability and hypothesis testing", "data analysis"), ("prompt engineering language models", "generative ai")]
        self.assertEqual(evaluate_hit(model, profiles), 1.0)

    def test_empty_fit_rejected(self):
        with self.assertRaises(ValueError):
            TfidfRecommender().fit([])


if __name__ == "__main__":
    unittest.main()
