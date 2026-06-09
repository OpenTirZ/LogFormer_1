<!-- 
  SEO Keywords: log anomaly detection, log anomaly detection using transformer, transformer log anomaly, 
  anomaly detection in system logs, log analysis deep learning, PyTorch log anomaly detection, 
  unsupervised log anomaly detection, HDFS log anomaly detection, GPT transformer log analysis, 
  log anomaly detection python, event log anomaly detection, transformer-based anomaly detection, 
  log mining, log monitoring AI, AIOps log anomaly, system log anomaly detection, 
  distributed systems log analysis, next event prediction log anomaly, log sequence anomaly detection,
  industrial log anomaly detection, deep learning for log analytics, log anomaly detection open source,
  LogFormer, LogFormer_1, OpenTirZ, Tirth Patel
-->

<p align="center">
  <img src="Logo.png" alt="LogFormer_1 — Transformer-Based Log Anomaly Detection System Logo" width="200"/>
</p>

<h1 align="center">LogFormer_1: Log Anomaly Detection using Transformers</h1>

<p align="center">
  <strong>A Lightweight Transformer-Based Log Anomaly Detection System for Industrial Log Analysis</strong>
</p>

<p align="center">
  <a href="https://github.com/OpenTirZ/LogFormer_1/releases/tag/v1.0">
    <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version 1.0.0"/>
  </a>
  <a href="https://github.com/OpenTirZ/LogFormer_1/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"/>
  </a>
  <img src="https://img.shields.io/badge/python-3.8%2B-yellow.svg" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/pytorch-2.0%2B-orange.svg" alt="PyTorch 2.0+"/>
  <img src="https://img.shields.io/badge/status-released-brightgreen.svg" alt="Released"/>
  <img src="https://img.shields.io/badge/anomaly_detection-transformer-blueviolet.svg" alt="Anomaly Detection Transformer"/>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#problem-statement">Problem</a> •
  <a href="#approach">Approach</a> •
  <a href="#model-architecture">Architecture</a> •
  <a href="#anomaly-detection">Anomaly Detection</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#results">Results</a> •
  <a href="#citation">Citation</a> •
  <a href="#future-work">Future Work</a>
</p>

---

## Overview

**LogFormer_1** is a lightweight Transformer-based **log anomaly detection** system designed for industrial log analysis and AIOps. The model learns normal log event patterns from historical system logs and identifies unusual behavior using next-event prediction — a technique adapted from language modeling to the domain of system log monitoring. By treating log events as tokens — similar to how language models process words in a sentence — LogFormer_1 learns the sequential structure of system behavior and flags deviations that may indicate failures, misconfigurations, or security attacks.

Built on a compact GPT-style Transformer architecture, LogFormer_1 is designed to be lightweight and trainable on Google Colab, making it accessible for researchers and practitioners working with industrial log data. The system operates in an **unsupervised** manner: it requires only normal log sequences for training and detects anomalies by measuring how unexpected observed events are relative to learned patterns. This makes LogFormer_1 a practical open-source solution for **anomaly detection in system logs**, **distributed systems monitoring**, and **AIOps log analytics**.

Whether you are researching **log anomaly detection using deep learning**, building a **log monitoring pipeline**, or exploring **transformer-based anomaly detection** for the first time, LogFormer_1 provides a clean, modular, and easy-to-understand codebase to get started.

## Problem Statement

Modern distributed systems generate massive volumes of log data every second. These logs are the primary source of operational intelligence — they record every significant system event, from block allocation and replication to error handling and exception reporting. However, manually sifting through millions of log lines to find anomalous behavior is not only impractical but also error-prone. The challenge of **log anomaly detection** — automatically identifying abnormal patterns in system logs — is one of the most critical problems in **AIOps** and **site reliability engineering**.

Traditional rule-based monitoring systems rely on predefined patterns and thresholds to flag issues. While effective for known failure modes, they fundamentally cannot detect unseen or novel failures — the very failures that are often the most critical. As systems grow in complexity, the gap between what rules can catch and what actually goes wrong continues to widen. **Unsupervised log anomaly detection** methods are needed that can generalize to previously unseen failure types without requiring labeled anomaly data.

LogFormer_1 addresses this challenge by learning the normal sequential behavior of log events directly from data. Instead of relying on handcrafted rules, it uses a **Transformer model** to capture the statistical patterns of normal log sequences and identifies anomalies as events that deviate significantly from these learned patterns. This approach generalizes to previously unseen failure types, making it far more robust than traditional rule-based log monitoring methods.

## Dataset

LogFormer_1 is evaluated on the **HDFS (Hadoop Distributed File System) Log Dataset** from [Loghub](https://github.com/logpai/loghub), a widely-used benchmark in the log analysis and log anomaly detection research community.

| Property | Details |
|----------|---------|
| **Dataset** | HDFS Log Dataset (from Loghub) |
| **Structured Log File** | `HDFS_2k.log_structured.csv` |
| **Template File** | `HDFS_2k.log_templates.csv` |
| **Event Templates** | E1 – E14 |
| **Source** | Loghub — A Large Collection of System Log Datasets for AI-driven Log Analytics |

Each `EventId` (E1 through E14) represents a specific system action such as:

- **Block allocation** — Assigning storage blocks to files
- **Block transfer** — Moving data blocks between nodes
- **Block replication** — Creating redundant copies for fault tolerance
- **Block deletion** — Releasing unused storage blocks
- **Block verification** — Integrity checks on stored data
- **Exception handling** — Logging errors, warnings, and recovery actions

The model treats these event IDs as discrete tokens in a vocabulary, learning the transitional probabilities between events in normal system operation. This tokenization approach enables the Transformer to process log sequences the same way a language model processes sentences — a core innovation that makes **log anomaly detection using transformers** both effective and elegant.

## Approach

LogFormer_1 follows a systematic pipeline from raw log data to anomaly scores, implementing a complete **log anomaly detection** workflow:

```
┌─────────────┐    ┌──────────────┐    ┌───────────────────┐    ┌────────────────┐
│  Raw Logs   │───▶│  Log Parsing  │───▶│  Event Tokenization│───▶│ Sliding Window │
└─────────────┘    └──────────────┘    └───────────────────┘    └───────┬────────┘
                                                                             │
                                                                             ▼
┌──────────────┐    ┌───────────────┐    ┌──────────────────┐    ┌────────────────┐
│ Anomaly Flag │◀───│ Anomaly Score │◀───│ Probability Calc  │◀───│  Transformer   │
└──────────────┘    └───────────────┘    └──────────────────┘    └────────────────┘
```

1. **Parse Logs** — Raw log messages are parsed into structured event IDs using a log parser (log parsing is the first step in any log anomaly detection pipeline).
2. **Tokenize Events** — Event IDs are mapped to numerical tokens for model consumption, converting log sequences into token sequences.
3. **Create Sequences** — Training sequences are generated using a sliding window over event histories, creating input-target pairs for next-event prediction.
4. **Train Transformer** — A GPT-style Transformer is trained to predict the next event in a sequence, learning normal log event transitions.
5. **Compute Probabilities** — During inference, the model outputs the probability distribution over possible next events given the context.
6. **Score Anomalies** — Anomaly scores are computed using negative log probability of the actual observed event.
7. **Flag Anomalies** — Events with high anomaly scores are flagged as potential anomalies in the log stream.

## Model Architecture

LogFormer_1 implements a compact GPT-style Transformer optimized for **log event prediction and anomaly detection**:

| Component | Details |
|-----------|---------|
| **Token Embedding** | Maps event IDs to dense vector representations (dim = 128) |
| **Positional Embedding** | Injects sequential position information into embeddings |
| **Multi-Head Self Attention** | 4 attention heads for capturing event dependencies and log sequence patterns |
| **Feed-Forward Network** | Two-layer MLP with GELU activation |
| **Layer Normalization** | Pre-norm architecture for stable training |
| **Prediction Head** | Linear projection to vocabulary size for next-event prediction |

### Configuration

| Hyperparameter | Value |
|----------------|-------|
| Vocabulary Size | 14 (E1–E14) |
| Context Length | 10 |
| Embedding Dimension | 128 |
| Attention Heads | 4 |
| Transformer Layers | 4 |
| Activation | GELU |

### Example — Next Event Prediction for Log Anomaly Detection

```
Input Sequence:  [E1] [E2] [E4] [E5]
                        │
                ┌───────▼───────┐
                │  Transformer   │
                │  (4 layers)    │
                └───────┬───────┘
                        │
                Predicted: [E6]
```

The model learns normal event transitions and predicts the most likely next event. If the actual event deviates significantly from the prediction, it receives a high anomaly score — indicating a potential anomaly in the log sequence.

## Anomaly Detection

LogFormer_1 uses a principled probabilistic approach to **anomaly detection in system logs**. During inference, the trained model computes the probability of each possible next event given the preceding context. The anomaly score for an observed event is defined as:

```
score = -log(P(actual_event | context))
```

- **Low score** → The event was expected given the context → **Normal behavior**
- **High score** → The event was unexpected given the context → **Potential anomaly**

This formulation naturally assigns higher anomaly scores to events that are rare or inconsistent with learned patterns, without requiring any labeled anomaly data during training. This unsupervised approach to **log anomaly detection** is particularly valuable in real-world scenarios where labeled anomalous logs are scarce or unavailable.

## Project Structure

```
LogFormer_1/
├── Attention/
│   └── MultiHeadAttention.py    # Multi-head self-attention mechanism
├── Data/
│   └── data.py                  # Data loading and preprocessing
├── Dataloader/
│   └── dataloader.py            # PyTorch DataLoader utilities
├── FeedForward/
│   └── FeedForward.py           # Position-wise feed-forward network
├── GELU/
│   └── GELU.py                  # GELU activation function
├── LayerNorm/
│   └── LayerNorm.py             # Layer normalization module
├── Testing/
│   ├── test.py                  # Anomaly scoring and evaluation
│   └── test.png                 # Anomaly score visualization
├── Transformer/
│   └── Transformer.py           # Transformer block composition
├── Logo.png                     # Project logo
├── main.py                      # Training entry point
└── training.log                 # Training logs with loss curves
```

## Getting Started

### Prerequisites

```bash
pip install torch pandas numpy matplotlib
```

### Training — Log Anomaly Detection Model

```bash
python main.py
```

This will:
- Load and preprocess the HDFS log dataset for anomaly detection
- Build training sequences using a sliding window
- Train the Transformer model with next-event prediction
- Save training logs to `training.log`

### Anomaly Detection — Inference and Scoring

```bash
python Testing/test.py
```

This will:
- Load the trained log anomaly detection model
- Calculate anomaly scores for test log sequences
- Generate visualization plots (saved as `Testing/test.png`)

### Quick Start on Google Colab

LogFormer_1 is designed to be lightweight enough to train on Google Colab, making **log anomaly detection with transformers** accessible to everyone:

1. Clone the repository: `git clone https://github.com/OpenTirZ/LogFormer_1.git`
2. Upload the HDFS dataset files to your Colab environment
3. Run `main.py` to train the model
4. Run `Testing/test.py` to evaluate and visualize anomaly scores

## Results

<p align="center">
  <img src="Testing/test.png" alt="Log Anomaly Detection Score Visualization — LogFormer_1 Anomaly Scores on HDFS Dataset" width="700"/>
</p>

<p align="center"><em>Anomaly score distribution across test log sequences — higher scores indicate detected anomalies</em></p>

The visualization above shows the anomaly scores computed for test log sequences using LogFormer_1. Peaks in the score indicate events that deviate significantly from the learned normal patterns — these are the **detected log anomalies**. This demonstrates the effectiveness of using a Transformer-based approach for **unsupervised log anomaly detection** on the HDFS dataset.

## Features

- **Transformer-Based Log Anomaly Detection** — Leverages self-attention for capturing long-range dependencies in log sequences
- **Lightweight & Efficient** — Trainable on Google Colab with minimal computational resources
- **Unsupervised Anomaly Detection** — Requires only normal log data for training; no labeled anomalies needed
- **Industrial Log Analysis** — Designed for real-world distributed system logs and AIOps workflows
- **Event-Level Anomaly Scoring** — Granular anomaly scores at the individual event level
- **Attention-Based Log Modeling** — Captures complex sequential patterns that rule-based systems miss
- **Next-Event Prediction** — GPT-style training objective for learning normal log event transitions
- **Open Source** — Fully open-source log anomaly detection system under MIT license

## Tech Stack

| Technology | Purpose |
|------------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) | Core language for log anomaly detection |
| ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white) | Deep learning framework for Transformer model |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) | Data manipulation and log preprocessing |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) | Numerical computing and array operations |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=python&logoColor=white) | Anomaly score visualization and plotting |

## Citation

If you use the HDFS dataset from [Loghub](https://github.com/logpai/loghub) in your research, please cite the following paper:

```bibtex
@inproceedings{zhu2023loghub,
  title={Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics},
  author={Zhu, Jieming and He, Shilin and He, Pinjia and Liu, Jinyang and Lyu, Michael R.},
  booktitle={IEEE International Symposium on Software Reliability Engineering (ISSRE)},
  year={2023}
}
```

> Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu. Loghub: A Large Collection of System Log Datasets for AI-driven Log Analytics. IEEE International Symposium on Software Reliability Engineering (ISSRE), 2023.

## Future Work

- **Attention Visualization** — Visualize attention weights to understand which past events influence predictions and improve interpretability of log anomaly detection results
- **Root Cause Analysis** — Extend the model to not only detect anomalies but also identify their root causes in the log stream
- **Failure Prediction** — Predict imminent failures before they occur based on early warning patterns in system logs
- **Real-Time Log Monitoring** — Deploy the model as a streaming service for live log monitoring and real-time anomaly detection
- **Larger Datasets** — Evaluate and scale to larger industrial log datasets (BGL, Thunderbird, Spirit) from Loghub
- **Robustness Testing** — Systematic evaluation against adversarial and distribution-shift scenarios in log anomaly detection

## Project Goal

Build a practical and efficient Transformer-based anomaly detection system that can learn system behavior from logs and automatically identify unusual events in large-scale distributed systems. LogFormer_1 represents the first step toward intelligent, automated **log anomaly detection** powered by modern deep learning and Transformer architectures.

---

## Author

<p align="center">
  <strong>Tirth Patel</strong><br/>
  <a href="https://github.com/Tirth9978">
    <img src="https://img.shields.io/badge/GitHub-@Tirth9978-181717?logo=github&logoColor=white" alt="GitHub @Tirth9978"/>
  </a>
</p>

---

<p align="center">
  <img src="Logo.png" alt="OpenTirZ — Open Source AI & Systems" width="60"/>
  <br/>
  <strong>OpenTirZ</strong> • Building Intelligent Systems
  <br/><br/>
  <a href="https://github.com/OpenTirZ/LogFormer_1">
    <img src="https://img.shields.io/github/stars/OpenTirZ/LogFormer_1?style=social" alt="Star LogFormer_1 on GitHub"/>
  </a>
  <a href="https://github.com/OpenTirZ/LogFormer_1">
    <img src="https://img.shields.io/github/forks/OpenTirZ/LogFormer_1?style=social" alt="Fork LogFormer_1 on GitHub"/>
  </a>
  <a href="https://github.com/OpenTirZ/LogFormer_1">
    <img src="https://img.shields.io/github/watchers/OpenTirZ/LogFormer_1?style=social" alt="Watch LogFormer_1 on GitHub"/>
  </a>
</p>

---

<details>
<summary>🔍 SEO Topics & Keywords</summary>

Log anomaly detection, anomaly detection in system logs, transformer log anomaly detection, deep learning log analysis, unsupervised log anomaly detection, HDFS log anomaly detection, GPT transformer log analysis, PyTorch log anomaly detection, log anomaly detection python, event log anomaly detection, transformer-based anomaly detection, log mining, log monitoring AI, AIOps log anomaly, system log anomaly detection, distributed systems log analysis, next event prediction log anomaly, log sequence anomaly detection, industrial log anomaly detection, deep learning for log analytics, log anomaly detection open source, LogFormer, LogFormer_1, OpenTirZ, Tirth Patel, log parsing, log template extraction, log event prediction, anomaly scoring, negative log likelihood anomaly detection, self-attention log analysis, GPT-style transformer logs, log anomaly detection GitHub, log anomaly detection repository, machine learning log anomaly, NLP for log analysis, sequence modeling log data, log anomaly detection research, log anomaly detection paper, AIOps, site reliability engineering, DevOps log monitoring, MLOps log analytics

</details>
