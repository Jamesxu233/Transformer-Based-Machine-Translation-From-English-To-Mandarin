# Transformer-Based Machine Translation: English to Mandarin

A reproducible, modular, and production-style implementation of a Transformer-based neural machine translation system for **English-to-Mandarin Chinese** translation.

This project demonstrates a complete deep learning pipeline for machine translation, including data preprocessing, vocabulary/tokenizer construction, Transformer model training, evaluation with standard MT metrics, and inference for custom sentences.

---

## Overview

This repository provides an end-to-end implementation of a Transformer-based sequence-to-sequence model inspired by the original **"Attention Is All You Need"** architecture. The project is designed to be:

- **Modular**: clear separation of data, model, training, evaluation, and inference logic
- **Reproducible**: configurable experiments and deterministic training setup
- **Maintainable**: production-style project structure and code organization
- **Demonstrative**: suitable for portfolio presentation, coursework refinement, and technical interviews

The system is built for **English → Mandarin Chinese** machine translation, but the overall framework can be extended to other language pairs.

---

## Key Features

- Transformer encoder-decoder architecture
- Config-driven training pipeline
- Modular data preprocessing and tokenization
- Support for training / validation / test splits
- Greedy decoding for inference
- BLEU-based evaluation
- Clean experiment logging and checkpoint saving
- Easily extensible for beam search, mixed precision, and larger pretrained tokenizers

---

## Repository Structure

```bash
transformer-en2zh/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── base_config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── samples/
├── notebooks/
│   └── exploratory_analysis.ipynb
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── seed.py
│   │   ├── io.py
│   │   └── logger.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tokenizer.py
│   │   ├── vocabulary.py
│   │   ├── dataset.py
│   │   └── collator.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── positional_encoding.py
│   │   ├── transformer.py
│   │   └── modules.py
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── inference.py
│   └── metrics/
│       ├── __init__.py
│       └── bleu.py
├── checkpoints/
├── outputs/
│   ├── predictions/
│   └── logs/
└── tests/
    ├── __init__.py
    └── test_shapes.py
```
---

## Project Workflow  

The project follows a standard machine learning lifecycle:  

1. **Prepare parallel translation data**  

2. **Clean and preprocess English / Mandarin text**  

3. **Build tokenizers and vocabularies**  

4. **Convert sentence pairs to indexed sequences**  

5. **Train Transformer model**  

6. **Evaluate on held-out validation/test data**  

7. **Run inference on custom English sentences**

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/transformer-en2zh.git
cd transformer-en2zh
```

### 2. Create environment

Using ```venv```:  

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```
---

## Dependencies

Typical dependencies include:  

- Python 3.10+

- PyTorch

- PyYAML

- NumPy

- pandas

- sacrebleu

- tqdm

- jieba

- matplotlib
---

## Data Format

The repository expects parallel corpora in a simple tabular or line-aligned format.  

Example:  

```bash
English:   I love machine learning.
Mandarin:  我喜欢机器学习。
```

Recommended processed CSV format:  

| **src_text** | **tgt_text** |
| --- | --- | 
| I love machine learning. | 我喜欢机器学习。 |
| How are you? | 你好吗？ |

Place raw files under:   

```bash
data/raw/
```

Processed files will be saved to:  

```bash
data/processed/
```
---

## Configuration

Training and model hyperparameters are defined in:  

```bash
configs/base_config.yaml
```

Example configurable fields:  

- data paths  

- batch size  

- max sequence length  

- embedding dimension  

- number of attention heads  

- number of encoder/decoder layers  

- dropout  

- optimizer settings  

- number of epochs  

- checkpoint path
---

## Usage

### 1. Prepare data

```bash
python scripts/prepare_data.py --config configs/base_config.yaml
```

### 2. Train the model

```bash
python scripts/train.py --config configs/base_config.yaml
```

### 3. Evaluate the model

```bash
python scripts/evaluate.py --config configs/base_config.yaml --split test
```

### 4. Predict custom translations

```bash
python scripts/predict.py --config configs/base_config.yaml --text "I enjoy studying neural networks."
```
---

## Example Results

Example translation behavior may look like:  

**Input**  

```bash
The weather is very nice today.
```

**Output**  

```bash
今天天气很好。
```

Evaluation can be reported using:  

- BLEU  

- token-level loss  

- validation perplexity  

**You may optionally add a results table here once experiments are finalized.**  

---

## Model Architecture

The implementation follows the Transformer encoder-decoder design with:  

- token embeddings

- sinusoidal positional encoding

- multi-head self-attention

- encoder-decoder cross-attention

- position-wise feed-forward layers

- residual connections + layer normalization
---

## Reproducibility

To improve reproducibility, the project includes:   

- centralized config management

- fixed random seeds

- checkpoint saving

- consistent preprocessing pipeline

- isolated scripts for train / eval / predict
---

## Future Improvements

Potential extensions include:  

- Beam search decoding

- Label smoothing

- Mixed precision training

- SentencePiece or BPE tokenization

- Pretrained multilingual tokenizers

- Attention visualization

- Experiment tracking with Weights & Biases

- Docker support

- CI-based testing
---

## Why This Project Matters

Machine translation is a representative sequence-to-sequence task that highlights several practical machine learning engineering skills:  

- NLP data preprocessing

- vocabulary/tokenizer design

- Transformer implementation

- training loop construction

- evaluation and inference design

- modular ML project organization

This repository is therefore intended not only as a translation system, but also as a demonstration of clean ML engineering practice.  

---

## License

This project is released under the MIT License.  

---

## Acknowledgments

This implementation is inspired by:  

- Vaswani et al., *Attention Is All You Need*

- Standard PyTorch sequence modeling practices

- Open-source neural machine translation engineering conventions
