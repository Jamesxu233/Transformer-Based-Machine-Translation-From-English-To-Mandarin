# Transformer-Based Machine Translation: English to Mandarin

A supervised neural machine translation project that implements an **encoder–decoder Transformer** in PyTorch to translate English sentences into Mandarin Chinese. The project covers parallel-corpus preprocessing, language-specific tokenization, padding and attention masks, teacher-forced training, autoregressive decoding, and BLEU/BERTScore evaluation.

**Course:** MSML 641 – Natural Language Processing  
**Institution:** University of Maryland, College Park  
**Team:** Jiayuan Shen, Ke Xu, Qiao Qin  
**Primary framework:** PyTorch

---

## 1. Project overview

The objective is to build an English-to-Mandarin translation model that generates Chinese translations while preserving the meaning of the English input. The system is trained with supervised sentence pairs from the Hugging Face dataset `magicsword/train-en-zh`.

The implementation follows the standard Transformer encoder–decoder design introduced in *Attention Is All You Need*, using PyTorch's `torch.nn.Transformer` module and custom token embeddings with sinusoidal positional encodings.

### Workflow

1. Load the English–Chinese parallel corpus and pretrained BERT tokenizers.
2. Split sentence pairs into English and Mandarin text.
3. Tokenize English at word/subword level and Mandarin at character-oriented token level using the selected BERT tokenizers.
4. Add `[CLS]` / `[SEP]` special tokens and convert tokens to vocabulary IDs.
5. Pad variable-length sequences and construct source padding masks, target padding masks, and a causal decoder attention mask.
6. Train the encoder–decoder Transformer with teacher forcing, AdamW, and cosine learning-rate scheduling.
7. Generate Mandarin translations autoregressively with greedy decoding.
8. Evaluate translation quality using SacreBLEU and BERTScore.

## 2. Dataset and tokenization

- **Dataset:** [`magicsword/train-en-zh`](https://huggingface.co/datasets/magicsword/train-en-zh)
- **Size stated in the project report:** 100,000 English–Chinese sentence pairs.
- **English tokenizer:** `bert-base-cased`
- **Chinese tokenizer:** `google-bert/bert-base-chinese`
- **Special tokens:** `[UNK]`, `[SEP]`, `[PAD]`, `[CLS]`, `[MASK]`

The source records use the `English sentence ||| Chinese translation` format. The preprocessing notebook separates the two languages, tokenizes each side, adds special tokens, maps tokens to integer IDs, and saves the processed examples as JSON.

The report describes English tokenization as word-level and Mandarin tokenization as character-level. More precisely, the actual segmentation is determined by the corresponding pretrained BERT tokenizer vocabularies.

## 3. Model architecture

The project implements a Transformer encoder–decoder using `torch.nn.Transformer`, with custom embedding layers that combine token embeddings and sinusoidal positional encodings.

| Parameter | Configuration |
|---|---:|
| Encoder layers | 3 |
| Decoder layers | 3 |
| Model / embedding dimension (`d_model`) | 512 |
| Attention heads | 8 |
| Dimension per head | 64 |
| Feed-forward dimension | 1,024 |
| Dropout | 0.1 |
| Maximum sequence length | 512 |

The report notes that the model is smaller than the original Transformer base configuration due to available compute resources.

### Sequence masks

- **Source padding mask:** prevents the encoder from attending to `[PAD]` positions.
- **Target padding mask:** prevents the decoder from attending to padded target tokens.
- **Causal target mask:** uses a triangular mask so that each decoder position can only attend to preceding and current target positions, not future tokens.
- **Memory padding mask:** prevents decoder cross-attention from using padded source positions.

### Training-time and inference-time decoding

- **Teacher forcing:** during training, the decoder receives the ground-truth target sequence shifted right.
- **Autoregressive generation:** during inference, the model feeds its previously generated token back into the decoder.
- **Decoding strategy:** greedy search (argmax); beam search is not implemented in this project.

## 4. Training configuration

The training notebook uses the following configuration:

| Hyperparameter | Value |
|---|---:|
| Learning rate | `8e-5` |
| Optimizer | AdamW |
| Learning-rate scheduler | CosineAnnealingLR |
| Epochs | 15 |
| Batch size | 128 |
| Training split | 99% |
| Validation split | 1% |

The training loop uses cross-entropy loss with padding tokens ignored. Model weights are saved as a PyTorch state dictionary.

## 5. Evaluation and reported results

The project evaluates translations with two complementary metrics:

- **SacreBLEU:** n-gram overlap between generated translations and reference translations.
- **BERTScore:** semantic similarity based on contextual token embeddings, reported as precision, recall, and F1.

The presentation's results table reports:

| Metric | Before training | After training |
|---|---:|---:|
| BLEU | 0.0 | 7.520 |
| BERTScore precision | 0.312 | 0.705 |
| BERTScore recall | 0.450 | 0.678 |
| BERTScore F1 | 0.365 | 0.691 |

The presentation also reports post-training BLEU n-gram precision values of 37.034, 11.786, 4.507, and 1.803.

**Result consistency note:** the written report's limitations section cites a BLEU score of 4.402, while the presentation's results table reports 7.520 after training. These values are inconsistent in the submitted materials. They are therefore both documented here rather than silently choosing one. Before using a single BLEU result on a résumé or in a formal report, rerun evaluation with the same checkpoint, validation split, SacreBLEU configuration, and aggregation settings, then record the reproducible result.

## 6. Example translations

The presentation/report includes these post-training examples:

| English input | Model output |
|---|---|
| `How old are you?` | `你几岁？` |
| `Are you worried about the African front?` | `你还有担心吗？` |

The examples show that the model can generate readable Mandarin outputs for some short inputs, but translation quality remains limited.

## 7. Repository structure

```text
.
├── README.md
├── requirements.txt
├── notebooks/
│   ├── datasetPreprocess.ipynb
│   ├── encoder_decoder.ipynb
│   └── train.ipynb
├── data/
│   └── README.md
├── models/
│   └── README.md
├── reports/
│   └── README.md
└── presentation/
    └── README.md
```

- `datasetPreprocess.ipynb`: downloads the Hugging Face dataset and tokenizers, preprocesses sentence pairs, and exports JSON.
- `encoder_decoder.ipynb`: defines positional embeddings and the Transformer model.
- `train.ipynb`: builds the Dataset/DataLoader, creates masks, trains and evaluates the model, and demonstrates translation generation.
- `data/`: destination for the generated preprocessed JSON file.
- `models/`: destination for the trained PyTorch state dictionary.
- `reports/` and `presentation/`: optional locations for sanitized report and slides.

The original project checklist mentions a `model_state_dict.pth` checkpoint, but that checkpoint was not among the files supplied for this repository update. Add it only if you have the actual trained weights and are permitted to redistribute them.

## 8. Environment setup

Recommended: Python 3.9–3.11. Install the dependencies:

```bash
pip install -r requirements.txt
```

The project uses:

- PyTorch
- Hugging Face `datasets` and `transformers`
- `evaluate`
- `sacrebleu`
- `bert-score`
- pandas, matplotlib, tqdm

A CUDA-enabled GPU is recommended for training. The code automatically selects CUDA when available, otherwise CPU.

## 9. How to run

Run the notebooks in this order. The commands below assume the current working directory is `notebooks/`.

### Step 1 — Preprocess the dataset

```bash
jupyter notebook datasetPreprocess.ipynb
```

Run all cells. The notebook writes:

```text
../data/processed_dataset.json
```

### Step 2 — Train and evaluate

Open:

```bash
jupyter notebook train.ipynb
```

Run the notebook from top to bottom. It loads `encoder_decoder.ipynb`, creates the model, trains for 15 epochs, evaluates before and after training, and saves the checkpoint to:

```text
../models/model_state_dict.pth
```

To load an existing checkpoint instead of training again, set `load_model = True` in the training notebook and ensure the checkpoint exists at the configured path.

### Step 3 — Translate a custom sentence

At the end of `train.ipynb`, edit `input_sentence` and run the generation cell:

```python
input_sentence = "Can you tell me the truth?"
output = generate_fn(input_sentence)
print(output)
```

## 10. Limitations

The original project report identifies two major limitations:

1. **Underfitting / limited translation quality:** the reported BLEU score is low, and the model has not overfit the training data. Potential improvements include more training epochs, a larger model, and hyperparameter tuning.
2. **Slow inference:** generation is autoregressive and the current implementation evaluates one sentence at a time (`batch_size=1`). The project uses greedy decoding rather than beam search.

Additional considerations for future work:

- Increase the validation set size and use a reproducible fixed split.
- Evaluate on a held-out test set, not only a small validation subset.
- Track validation metrics during training and use checkpoint selection.
- Investigate subword tokenization, sequence-length filtering, and vocabulary alignment.
- Optimize batched decoding and consider beam search.
- Record exact package versions, random seeds, and checkpoint provenance.

## 11. References

- Vaswani et al., *Attention Is All You Need* (2017): https://arxiv.org/abs/1706.03762
- Hugging Face dataset: https://huggingface.co/datasets/magicsword/train-en-zh
- English tokenizer: https://huggingface.co/bert-base-cased
- Chinese tokenizer: https://huggingface.co/google-bert/bert-base-chinese
- PyTorch Transformer documentation: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
- SacreBLEU: https://github.com/mjpost/sacrebleu
- BERTScore: https://github.com/Tiiiger/bert_score

## 12. Academic context

This project was completed by Jiayuan Shen, Ke Xu, and Qiao Qin for the Natural Language Processing course at the University of Maryland, College Park. Preserve the original team attribution when redistributing the course project.
