# Sentiment Analysis on Restaurant Reviews

## Overview  
This project classifies restaurant reviews as either **positive** or **negative** using machine learning and deep learning techniques. The dataset contains labeled restaurant reviews with balanced sentiment classes. We experimented with multiple models to compare the performance of classical ML algorithms against state-of-the-art deep learning models.

We implemented and evaluated three models:

- Logistic Regression  
- Random Forest  
- Fine-tuned BERT (Transformer)  

Evaluation metrics include **accuracy**, **F1-score**, and **confusion matrices**. We applied text preprocessing and feature engineering techniques like **TF-IDF vectorization** and **contextual embeddings** using BERT.

---

## Files in the Repository

### 1. `Sentiment Analysis on Restaurant Reviews.ipynb`
Contains complete code for:

#### Dataset Preprocessing:
- Text cleaning (tokenization, lowercasing)
- TF-IDF vectorization with up to 5000 features
- BERT tokenization for transformer-based input
- Train-test split (80% train, 20% test, stratified)
  
#### Feature Engineering:
- TF-IDF word embeddings
- Contextual embeddings via Hugging Face BERT

#### Model Training:
- Logistic Regression (with hyperparameter tuning)
- Random Forest (with hyperparameter tuning)
- Fine-tuned BERT model

#### Evaluation & Visualization:
- Accuracy and F1-score computation
- Confusion matrix for all models
- Word clouds for positive and negative reviews

---

### 2. `Sentiment Analysis on Restaurant Reviews Report.pdf`
A detailed project report including:

- Motivation and use-case of sentiment analysis in food services
- Dataset description and balance details
- Overview of preprocessing and feature engineering strategies
- Description of each model and hyperparameter optimization
- Comparative evaluation and insights
- Conclusions and suggested future work

---

## Project Workflow

### Step 1: Dataset Preprocessing
- Text normalization and cleaning
- Tokenization (TF-IDF & BERT-based)
- Feature vector generation
- Stratified splitting into training and testing sets

### Step 2: Model Training
Train and compare the following models:
- **Logistic Regression** (baseline classifier)
- **Random Forest** (ensemble classifier)
- **BERT** (fine-tuned transformer)

### Step 3: Hyperparameter Tuning
- `RandomizedSearchCV` applied for Logistic Regression and Random Forest
- BERT fine-tuned using transformer pipelines and GPU acceleration

### Step 4: Evaluation
- Evaluation using Accuracy and F1-Score
- Visualization using Confusion Matrices

---

## Results Summary

| Model               | Train Acc | Train F1 | Test Acc | Test F1 |
|--------------------|-----------|----------|----------|---------|
| Logistic Regression| 99.04%    | 0.9908   | 94.23%   | 0.9434  |
| Random Forest      | 97.12%    | 0.9722   | 84.62%   | 0.8519  |
| BERT               | 99.52%    | 0.9953   | **96.15%** | **0.9615** |

---

## Key Insights

- **BERT** provided the highest test accuracy and F1-score due to its contextual understanding of language.
- **Logistic Regression** performed strongly with TF-IDF features, showing simple models can be effective.
- **Random Forest** showed signs of overfitting, performing well on training data but underperforming on test data.
- TF-IDF remains a robust method for feature extraction when deep learning is not feasible.

---
