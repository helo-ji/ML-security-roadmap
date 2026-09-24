# ML Security Roadmap

A hands-on learning roadmap focused on Machine learning, Cybersecurity, Adversarial Machine learning, and Security-focused AI.

## Goal

Build the foundations needed to work on practical Machine Learning and Cybersecurity projects.

The roadmap progresses from Python and Machine Learning fundamentals into:

- Adversarial Machine Learning
- Network Intrusion Detection
- Retrieval-Augmented Generation(RAG)
- Local Security AI

## Prerequisites

The prerequisite labs build the foundation needed for the main projects.

- [x] Lab 0 — Python Fundamentals
- [x] Lab 1 — NumPy Fundamentals
- [ ] Lab 2 — Pandas Fundamentals
- [ ] Lab 3 — Visualization and EDA
- [ ] Lab 4 — Machine Learning Fundamentals
- [ ] Lab 5 — Classification Evaluation

## Project 1 — Adversarially Robust Network Intrusion Detection

Build a network intrusion detection system and investigate adversarial attacks against machine-learning-based security systems.

Topics include:

- Dataset exploration
- Data preprocessing
- Data leakage
- Baseline models
- Class imbalance
- Error analysis
- Adversarial examples
- FGSM
- PGD
- Attack strength
- Adversarial training
- Robustness evaluation

## Project 2 — Local Security RAG Assistant

Build a local security-focused Retrieval-Augmented Generation system.

Topics include:

- RAG fundamentals
- Document collection
- Text extraction
- Chunking
- Embeddings
- Semantic retrieval
- BM25
- Hybrid retrieval
- Reranking
- Local LLMs
- Citations
- Hallucination testing
- Evaluation
- Docker
- Privacy considerations

## Repository Structure

```text
ML-security-roadmap/
├── .gitignore
├── README.md
├── requirements.txt
├── .venv/
│
├── pre-requisites/
│   ├── lab0_python/
│   ├── lab1_numpy/
│   ├── lab2_pandas/
│   ├── lab3_eda/
│   ├── lab4_ml/
│   └── lab5_evaluation/
│
├── project1/
│
└── project2/