# Machine Learning Analysis

A progression through machine learning from the ground up - classical models, neural networks, and transformer-based NLP. 

---

## What's Inside

### 1. Classical Machine Learning
**Folder:** `classification_regression/`

Regression and classification experiments covering the full workflow: data preprocessing, feature scaling, train-test splitting, and model evaluation. Metrics used include Accuracy and R² depending on the task.

Datasets are in the `datasets/` directory.

---

### 2. Neural Networks
**Folder:** `neural_networks/`

Two implementations:

**Feedforward MLP** - multi-layer perceptron for classification, covering layer design, loss functions, and performance evaluation.

**Time Series Forecasting** - LSTM-based model for sequential prediction, including sequence creation, training, and visualization of results.

---

### 3. Transformer-Based NLP
**Folder:** `transformers/`

#### Sentiment Analysis
**Subfolder:** `transformers/sentiment_analysis/`

A pretrained transformer fine-tuned on drug review data for binary sentiment classification. Uses the Hugging Face Trainer API with accuracy and F1-score evaluation.

Pipeline: rating-to-label mapping → tokenization → fine-tuning → evaluation.

#### Text Generation
**Subfolder:** `transformers/text_gen/`

An interactive terminal script that takes user input and generates a natural continuation using a pretrained causal language model (Qwen). Generation is controlled via temperature, top-p sampling, and repetition penalty to balance creativity and coherence.

```bash
python transformer_text_generation.py
```

```
Ready. Type a sentence and press Enter. Type 'q' to quit.
You: green curtains are much better than
Model: green curtains are much better than blue ones, especially in terms
of aesthetic appeal. Green is known for its calming effects...
```

---

## Tech Stack

Python · scikit-learn · PyTorch · Hugging Face Transformers · LSTM · Qwen
