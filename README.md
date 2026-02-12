# Machine Learning Analysis

This repository contains implementations of multiple machine learning approaches,
progressing from classical methods to neural networks and transformer-based models.

The project demonstrates understanding of:

- Classical Machine Learning (Regression & Classification)
- Neural Networks (MLP and Time Series models)
- Transformer Fine-Tuning for NLP
- Large Language Model Inference for Text Generation


---

---

## 1. Classical Machine Learning

Folder: `classification_regression/`

Includes:
- Classification experiments
- Regression experiments (multivariate)

Key concepts covered:
- Data preprocessing
- Feature scaling
- Train-test splitting
- Model evaluation (Accuracy, R², etc.)

Datasets are located in the `datasets/` directory.

---

## 2. Neural Networks

Folder: `neural_networks/`

Includes:
- Feedforward neural network for classification
- Time series forecasting model

Topics covered:
- Multi-Layer Perceptron (MLP)
- Sequence creation for time series
- LSTM-based forecasting 
- Loss functions
- Performance evaluation and visualization

---

## 3. Transformer-Based NLP

Folder: `transformers/`

### 3.1 Sentiment Analysis

Subfolder: `transformers/sentiment_analysis/`

A pretrained Transformer model is fine-tuned on drug review data
to perform binary sentiment classification.

Process:
- Rating into Sentiment label mapping
- Tokenization with pretrained tokenizer
- Fine-tuning using Hugging Face Trainer API
- Evaluation using Accuracy and F1-score

---

### 3.2 Text Generation

Subfolder: `transformers/text_gen/`

An interactive text generation script using a pretrained
causal language model (Qwen).
Limited to smaller versions with respect to hardware capabilities.

The script:
- Accepts user input from the terminal
- Generates a natural continuation
- Uses sampling techniques (temperature, top-p, repetition penalty)

Generation parameters control creativity and coherence.

Once started, expected output (with input):

Ready. Type a sentence and press Enter. Type 'q' to quit. 

You: green curtains are much better than

Model: green curtains are much better than blue ones, especially in terms of aesthetic appeal! Green is known for its calming effects that blue lacks due to its ability to absorb some UV rays. While both can create an inviting atmosphere, green offers subtle hints of nature without overpoweeen is known for its calming effects that blue lacks due to its ability to absorb some UV rays. While boring it with too many colors. If you're looking for something elegant but not overwhelming, blue hints of nature without overpowering it with too many colors. If t not overwhelming, blue might be your 
might be your best choice among these options. Enjoy your space! 

```bash
python transformer_text_generation.py


