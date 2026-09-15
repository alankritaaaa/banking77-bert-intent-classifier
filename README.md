# 🏦 Banking77 BERT Intent Classifier

A BERT-based NLP project for automatically classifying banking customer-service queries into **77 fine-grained intents** using the Banking77 dataset.

The project fine-tunes the pretrained **BERT (`bert-base-uncased`)** model using PyTorch and Hugging Face Transformers, covering the complete workflow from data preparation and tokenization to model training, evaluation, error analysis, and model persistence.

---

## 📌 Project Overview

Banking applications receive a large number of customer-service queries such as:

- "My card hasn't arrived yet."
- "Why was I charged twice?"
- "I don't recognize this payment."
- "My transfer is still pending."
- "Why hasn't my refund arrived?"

Although these queries are short, they can correspond to different support workflows.

This project uses **BERT** to understand the context of each query and classify it into one of **77 predefined banking intents**.

### Example

**Input:**
```text
Why hasn't my refund arrived?
```

**Predicted Intent:**
```text
Refund_not_showing_up
```

The goal is to build a reliable intent classifier that could serve as the NLP component of an automated customer-support or ticket-routing system.

---

## 🎯 Problem Statement

This is a multi-class text classification problem.

Given a banking-related customer query, the model must predict exactly one intent from 77 possible categories.

```text
Customer Query
      ↓
BERT Tokenizer
      ↓
BERT Encoder
      ↓
Classification Head
      ↓
77 Intent Scores
      ↓
Predicted Intent
```

The model therefore needs to learn the semantic differences between closely related banking queries rather than simply matching keywords.

---

## 📊 Dataset

The project uses the Banking77 dataset, a dataset designed for fine-grained intent classification in the banking domain.

### Dataset Statistics

| Property | Value |
|---|---|
| Total examples | 13,083 |
| Original training examples | 10,003 |
| Test examples | 3,080 |
| Number of intents | 77 |
| Training subset | 9,002 |
| Validation subset | 1,001 |
| Test set | 3,080 |

The original training set is further divided into training and validation subsets using a 90/10 stratified split with `random_state=42`.

The official test set is kept completely separate and untouched during training/model selection, and used only for final evaluation.

---

## 🏷️ Intent Classification

The dataset contains 77 banking-related intent categories.

Examples include:

```text
card_arrival
card_linking
card_not_working
card_payment_not_recognised
cash_withdrawal
cash_withdrawal_not_recognised
cash_withdrawal_charge
declined_card_payment
declined_transfer
failed_transfer
pending_card_payment
Refund_not_showing_up
activate_my_card
change_pin
lost_or_stolen_card
lost_or_stolen_phone
exchange_rate
exchange_via_app
```

The complete set of labels is generated directly from the dataset and mapped to numerical class IDs for model training.

---

## 🤖 Model

The project uses:

```text
bert-base-uncased
```

with a BERT sequence-classification head:

```text
BertForSequenceClassification
```

The final classification layer contains:

```text
77 output classes
```

### Architecture

```text
                Customer Query
                      ↓
               BERT Tokenizer
                      ↓
            Token IDs + Attention Mask
                      ↓
               BERT Encoder
                      ↓
          Contextual Representation
                      ↓
           Classification Head
                      ↓
                77 Classes
                      ↓
             Predicted Intent
```

---

## 🧠 Why BERT?

Traditional text-classification approaches such as keyword matching or TF-IDF primarily rely on word-level features.

BERT instead produces contextual representations of text.

For example, the meaning of:

```text
"payment"
```

can depend on the words surrounding it.

BERT's Transformer architecture allows the model to consider relationships between tokens when constructing the representation of a query.

This makes it suitable for distinguishing between semantically similar banking intents.

---

## 🔤 Tokenization

Before the text is passed to BERT, it is converted into tokens using the pretrained BERT tokenizer:

```python
BertTokenizer.from_pretrained("bert-base-uncased")
```

The project uses:

- Maximum sequence length: 64 tokens

Longer sequences are truncated, while shorter sequences are padded.

The tokenizer produces the model inputs required by BERT, including:

```text
input_ids
attention_mask
```

---

## 🔢 Label Encoding

The textual intent labels are converted into numerical IDs.

Two mappings are created:

```text
label2id
id2label
```

For example:

```text
Intent name
    ↓
Numerical class ID
```

This allows the neural network to train using numerical targets while preserving human-readable intent names for predictions and evaluation.

---

## 🏋️ Training

The pretrained BERT model is fine-tuned on the Banking77 training data.

During fine-tuning, the model learns how general language representations from pretrained BERT correspond to banking-specific customer intents.

### Training process

```text
Pretrained BERT
      ↓
Banking77 Training Data
      ↓
Tokenization
      ↓
Forward Pass
      ↓
Prediction
      ↓
Loss Calculation
      ↓
Backpropagation
      ↓
Parameter Updates
      ↓
Improved Intent Classification
```

The model was trained for 3 epochs.

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Base model | bert-base-uncased |
| Number of classes | 77 |
| Epochs | 3 |
| Training batch size | 8 |
| Evaluation batch size | 8 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Maximum sequence length | 64 |
| Validation split | 10% |
| Random seed | 42 |
| Evaluation strategy | Every epoch |
| Best model metric | Weighted F1 (on validation set) |

The model uses Hugging Face's `Trainer` API for the training loop and evaluation workflow.

---

## 📈 Evaluation

The project evaluates the classifier using:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification report
- Confusion matrix

F1 score is calculated using the weighted average, which accounts for the contribution of each class according to its number of examples.

---

## 📊 Validation Performance (during training)

The validation performance improved across the three training epochs. This validation set was used to select the best checkpoint during training — **not** as the final reported result (see Final Test Results below for that).

| Epoch | Training Loss | Validation Loss | Validation Accuracy | Validation Weighted F1 |
|---|---|---|---|---|
| 1 | 1.7717 | 1.5077 | 79.12% | 77.04% |
| 2 | 0.8150 | 0.7011 | 88.51% | 88.31% |
| 3 | 0.5643 | 0.5432 | 90.11% | 89.92% |

The best checkpoint (by validation weighted F1) was selected at epoch 3.

---

## ✅ Final Test Results

The numbers above are from the **validation set**, which was also used to pick the best checkpoint during training — so it's expected to look slightly optimistic. The number that actually reflects real-world generalization is performance on the **held-out test set** (3,080 examples, never used in training or model selection):

| Metric | Score |
|---|---|
| **Test Accuracy** | **89.61%** |
| **Test Weighted F1** | **89.06%** |
| **Test Macro F1** | 89.06% |

(Macro and weighted F1 come out identical here because the test set is perfectly class-balanced — exactly 40 examples per intent.)

This is a small drop from the validation numbers, which is normal and healthy — a much larger gap would have suggested overfitting to the validation set during checkpoint selection.

---

## 🔍 Error Analysis

A detailed classification report is generated for all 77 intents, covering precision, recall, F1-score, and support for each individual intent.

This makes it possible to identify classes that the model handles well and classes that are more difficult to distinguish. As expected, semantically similar intents involving cards, transfers, refunds, payments, and cash withdrawals are more challenging for the classifier than clearly distinct ones.

**The clearest concrete example of this: `virtual_card_not_working` scored 0.00 precision, 0.00 recall, and 0.00 F1 — the model never correctly predicted this intent on the test set.** Looking at the confusion matrix, this appears to be genuine confusion with two closely related virtual-card intents:

| Intent | Precision | Recall | F1 |
|---|---|---|---|
| `virtual_card_not_working` | 0.00 | 0.00 | 0.00 |
| `get_disposable_virtual_card` | 0.58 | 0.75 | 0.65 |
| `getting_virtual_card` | 0.59 | 0.98 | 0.74 |

All three intents revolve around virtual cards, and with only 40 training examples per class, the model struggled to separate "my virtual card isn't working" complaints from "how do I get a virtual card" requests. This is a genuine, specific failure worth investigating further (see Future Improvements), not just a general "similar intents are harder" statement.

By contrast, several intents were classified perfectly (F1 = 1.00), including `age_limit`, `apple_pay_or_google_pay`, `passcode_forgotten`, `verify_source_of_funds`, and `verify_top_up` — these tend to use more distinctive vocabulary with less overlap with other categories.

The full report is available in:

```text
results/classification_report.txt
```

---

## 📉 Confusion Matrix

A confusion matrix is generated to analyze the model's predictions across all 77 intent categories.

It provides a class-by-class view of:

- correct predictions
- incorrect predictions
- frequently confused intents

The matrix is saved as:

```text
results/banking77-confusion-matrix.png
```

### Interpretation

```text
Rows    → Actual intent
Columns → Predicted intent
```

A strong diagonal indicates that the model is correctly classifying a large number of examples.

Off-diagonal values reveal which intents are being confused with one another — most visibly the virtual-card cluster described above.

---

## 💾 Model Persistence

After training, the model and tokenizer are saved using Hugging Face's `save_pretrained()` functionality.

```python
model.save_pretrained("banking77-bert-final")
tokenizer.save_pretrained("banking77-bert-final")
```

This allows the trained model to be reused later without retraining from scratch.

The trained model files are not included in this repository because the model weights are large (~440MB).

---

## 📁 Project Structure

```text
banking77-bert-intent-classifier/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── banking77_bert_classifier.ipynb
│
└── results/
    ├── classification_report.txt
    └── banking77-confusion-matrix.png
```

**`notebooks/`** — contains the complete implementation notebook covering: dataset loading, data preparation, label encoding, train/validation split, tokenization, PyTorch dataset creation, BERT initialization, model training, evaluation, predictions, classification report, confusion matrix, and model saving.

**`results/`** — contains the generated evaluation artifacts.

**`requirements.txt`** — lists the Python dependencies required to reproduce the project.

**`.gitignore`** — prevents large model files, checkpoints, datasets, virtual environments, and temporary files from being committed.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | BERT model and training |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Data splitting and evaluation |
| Matplotlib | Visualization |
| Seaborn | Confusion matrix visualization |
| Jupyter / Google Colab | Development and model training |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/banking77-bert-intent-classifier.git
cd banking77-bert-intent-classifier
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
```
On Windows:
```bash
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Open the Notebook
```bash
jupyter notebook
```
Then open:
```text
notebooks/banking77_bert_classifier.ipynb
```

Run the notebook cells in order to reproduce the preprocessing, training, and evaluation workflow.

---

## ⚠️ Limitations

**Fixed Intent Set** — the classifier can only predict one of the 77 intents present in the Banking77 dataset. It cannot automatically create or discover a new intent.

**Domain Specificity** — the model is fine-tuned specifically for banking customer-service queries and may not generalize well to unrelated domains.

**Similar Intent Categories** — some banking intents are highly similar, making certain examples difficult to distinguish (see Error Analysis above for the concrete `virtual_card_not_working` case).

**Computational Cost** — BERT requires considerably more computational resources than traditional machine-learning classifiers.

---

## 🔮 Future Improvements

Several extensions could turn this project into a more complete production-oriented system.

### 1. Inference Script
Add a simple script that accepts a customer query and returns the predicted intent.
```text
Input:  "My card hasn't arrived."
Output: card_arrival
```

### 2. Confidence Scores
Return the predicted intent along with its confidence score.
```text
Intent: card_arrival
Confidence: 94.2%
```
Low-confidence predictions could then be routed to human support.

### 3. Fix the Virtual Card Confusion
Targeted error analysis and possibly data augmentation specifically for `virtual_card_not_working`, `get_disposable_virtual_card`, and `getting_virtual_card` — the one clear, identified weak point in the current model.

### 4. Model Comparison
Compare BERT with other Transformer architectures such as DistilBERT, RoBERTa, or DeBERTa.

### 5. API Deployment
Expose the trained model through an API using a framework such as FastAPI.
```text
Client → FastAPI → BERT Intent Classifier → Predicted Intent
```

### 6. Automated Support Routing
Integrate the classifier into a larger support-ticket routing pipeline.
```text
Customer Query → Intent Classifier → Predicted Department/Workflow → Automated Ticket Routing
```

---

## 💡 Key Concepts Demonstrated

This project demonstrates practical implementation of:

- Transformer-based NLP
- BERT and self-attention-based language representations
- Transfer learning and fine-tuning
- Text tokenization
- Multi-class classification
- PyTorch datasets
- Hugging Face Transformers
- Model evaluation: precision, recall, and F1
- Confusion matrix analysis
- The distinction between validation and test performance
- Model persistence
- Reproducible ML workflows

---

## 📚 What I Learned

Through this project, I gained hands-on experience with the complete lifecycle of a Transformer-based NLP classification task.

The project helped me understand how to:

- work with a real-world NLP dataset
- prepare labeled text data
- create train/validation splits
- tokenize text for BERT
- convert textual labels into model-compatible IDs
- build a PyTorch dataset
- fine-tune a pretrained Transformer
- evaluate a multi-class classifier correctly (and why validation and test numbers should be reported separately)
- analyze class-level performance and identify specific, named failure cases rather than general statements
- interpret a confusion matrix
- save and reload trained model artifacts

This project also provides a foundation for moving toward more advanced NLP systems such as semantic search, RAG pipelines, and LLM-based applications.

---

## 📌 Conclusion

This project demonstrates an end-to-end BERT-based intent classification pipeline for banking customer-service queries.

Starting with raw text, the system performs:

```text
Data Loading
     ↓
Preprocessing
     ↓
Train / Validation Split
     ↓
Tokenization
     ↓
BERT Fine-Tuning
     ↓
77-Class Classification
     ↓
Evaluation
     ↓
Error Analysis
     ↓
Model Saving
```

The model reached **89.61% accuracy and 89.06% weighted F1 on the held-out test set** after three epochs of fine-tuning — with one clearly identified weak point (`virtual_card_not_working`, confused with related virtual-card intents) documented for future improvement rather than glossed over.

The project demonstrates how pretrained Transformer models can be adapted to practical, domain-specific NLP problems and provides a foundation for building automated customer-support and ticket-routing systems.

---

## 🙏 Acknowledgements

This project uses the Banking77 dataset for banking intent classification.

The model implementation uses Hugging Face Transformers and PyTorch.
