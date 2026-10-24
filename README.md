# Learning Resource Recommender

A transparent content-based recommendation system that matches learner profiles to relevant AI, machine-learning, and data-science resources using TF-IDF and cosine similarity.

## Overview

Learners often know a goal but not which resource to start with. This project represents each resource as a text document, builds TF-IDF vectors, and ranks resources by similarity to a natural-language learner profile.

## Problem statement

Given a short description of a learner's interests and goals, recommend the most relevant educational resources while supporting an optional skill-level filter.

## Features

- TF-IDF feature representation implemented from scratch.
- Cosine-similarity ranking.
- Optional beginner, intermediate, or advanced filtering.
- Small labeled resource catalog and evaluation helper.
- CLI, tests, and standard-library-only implementation.

## Technology stack

- Python 3.10+
- Information retrieval: TF-IDF and cosine similarity
- CSV, `argparse`, `dataclasses`, and `math`
- `unittest` for testing

## Installation

```bash
git clone https://github.com/mohan843/learning-resource-recommender.git
cd learning-resource-recommender
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

No third-party runtime dependencies are required.

## Execution

```bash
python recommender.py --profile "I want to learn tokenization and text classification"
```

Filter by level:

```bash
python recommender.py --profile "neural networks and tensors" --level advanced --top-k 2
```

## Testing

```bash
python -m unittest discover -s tests -v
```

## Sample output

```text
Recommended resources:
  Natural Language Processing Basics [beginner] — similarity 0.478
  Deep Neural Networks [advanced] — similarity 0.291
  Practical Machine Learning [intermediate] — similarity 0.143
```

Scores depend on the catalog contents and query wording.

## Limitations and responsible use

This is a small educational catalog, not a complete curriculum planner. Similarity ranking can reflect wording rather than true pedagogical quality. Learners should consider prerequisites, course quality, accessibility, and instructor guidance.

## Future improvements

- Add learner feedback and implicit interaction signals.
- Compare content-based ranking with collaborative filtering.
- Add prerequisite graphs and diversity-aware recommendations.
- Support embeddings for semantic similarity beyond shared words.
- Build a web interface with progress tracking.
