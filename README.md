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
- [x] Lab 2 — Pandas Fundamentals
- [x] Lab 3 — Visualization and EDA
- [x] Lab 4 — Machine Learning Fundamentals
- [x] Lab 5 — Classification Evaluation

## Project 1 — Adversarially Robust Network Intrusion Detection

Build a machine-learning-based network intrusion detection prototype that analyses network-traffic records and classifies them as normal or suspicious.

The initial MVP focuses on training a model, evaluating its predictions, and providing a simple interface for analysing traffic records.

Topics include:
- Dataset exploration
- Data preprocessing
- Train/test splitting
- Baseline models
- Class imbalance
- Classification metrics
- Error analysis
- Interactive Streamlit interface
- Reproducible model training
- Testing and GitHub Actions
- Planned Security Extension

Extend the detector to investigate the robustness of machine-learning-based security systems against adversarial attacks.

Planned topics include:
- Adversarial examples
- Fast Gradient Sign Method (FGSM)
-Projected Gradient Descent (PGD)
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
## Future Project Ideas

Alternative projects to explore after the current roadmap projects.

### Future Idea 1 — Security Log Analyser

Build a Python tool that analyses authentication logs and identifies suspicious patterns, such as repeated failed login attempts and unusual activity.

**Topics:**
- Log parsing and pattern detection
- Python automation
- Security monitoring and SOC workflows
- Alerts and reporting
- DevOps fundamentals



### Future Idea 2 — Secure Python API with CI Security Checks

Build a small Python API with Docker, automated tests, and security checks integrated into a CI pipeline.

**Topics:**
- REST API fundamentals
- Input validation and secure coding
- Automated testing
- Docker
- CI pipelines
- Dependency and code security checks
- DevSecOps fundamentals



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
