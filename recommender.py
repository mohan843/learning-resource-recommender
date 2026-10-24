"""Content-based learning resource recommendations with TF-IDF cosine similarity."""
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Resource:
    title: str
    topic: str
    description: str
    level: str

    @property
    def document(self) -> str:
        return f"{self.topic} {self.description} {self.level}"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_resources(path: str | Path) -> list[Resource]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [Resource(row["title"], row["topic"], row["description"], row["level"]) for row in csv.DictReader(handle)]


class TfidfRecommender:
    def fit(self, resources: Sequence[Resource]) -> "TfidfRecommender":
        if not resources:
            raise ValueError("At least one resource is required")
        self.resources = list(resources)
        tokenized = [tokenize(resource.document) for resource in resources]
        vocabulary = sorted(set(token for document in tokenized for token in document))
        self.vocabulary = vocabulary
        document_frequency = {term: sum(term in document for document in tokenized) for term in vocabulary}
        self.idf = {term: math.log((1 + len(resources)) / (1 + frequency)) + 1 for term, frequency in document_frequency.items()}
        self.matrix = [self._vector(document) for document in tokenized]
        return self

    def _vector(self, tokens: Sequence[str]) -> dict[str, float]:
        counts = {term: tokens.count(term) for term in set(tokens)}
        total = len(tokens) or 1
        return {term: (count / total) * self.idf[term] for term, count in counts.items() if term in self.idf}

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        dot = sum(value * right.get(term, 0.0) for term, value in left.items())
        norm_left = math.sqrt(sum(value * value for value in left.values()))
        norm_right = math.sqrt(sum(value * value for value in right.values()))
        return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0

    def recommend(self, profile: str, top_k: int = 3, level: str | None = None) -> list[tuple[Resource, float]]:
        query = self._vector(tokenize(profile))
        scored = [(resource, self._cosine(query, vector)) for resource, vector in zip(self.resources, self.matrix)]
        if level:
            scored = [(resource, score) for resource, score in scored if resource.level.lower() == level.lower()]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]


def evaluate_hit(recommender: TfidfRecommender, profiles: Sequence[tuple[str, str]]) -> float:
    hits = 0
    for profile, expected_topic in profiles:
        recommendations = recommender.recommend(profile, 1)
        hits += bool(recommendations and expected_topic.lower() in recommendations[0][0].topic.lower())
    return hits / len(profiles) if profiles else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommend learning resources from a learner profile.")
    parser.add_argument("--data", default="data/resources.csv")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--level")
    args = parser.parse_args()
    model = TfidfRecommender().fit(load_resources(args.data))
    print("Recommended resources:")
    for resource, score in model.recommend(args.profile, args.top_k, args.level):
        print(f"  {resource.title} [{resource.level}] — similarity {score:.3f}")
