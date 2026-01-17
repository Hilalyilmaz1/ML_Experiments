# 🧠 WordMorph: Text Simplification & Model Robustness Analysis

WordMorph is an accessibility-focused project originally designed to help individuals with dyslexia read texts more easily by transforming complex sentences into simpler, more readable forms.

This repository extends that idea with an experimental question:

> **Does text simplification negatively affect a machine learning model’s ability to learn, or can meaning-preserving simplification be safely applied without performance loss?**

To answer this, we compare traditional frequency-based models with context-aware transformer models on both original and simplified texts.

---

## 🎯 Motivation

Text simplification is commonly assumed to reduce information density.  
From a machine learning perspective, this raises an important concern:

- Does simplification cause **loss of contextual information**?
- Will a model trained on simplified text perform worse?
- Are modern NLP models robust to surface-level lexical changes?

These questions are especially important for **accessibility-oriented NLP systems**, where human readability and model performance must coexist.

---

## 📊 Dataset

- **Source:** Hugging Face Datasets
- **Dataset:** `BayanDuygu/TrGLUE` (SST-2 configuration)
- **Language:** Turkish
- **Task:** Binary sentiment classification (positive / negative)

The dataset consists of short sentences, making it suitable for controlled experiments on lexical vs contextual understanding.

---

## 🧪 Experimental Setup

The experiments were conducted in three main stages.

---

### 1️⃣ Baseline Model: TF-IDF + Logistic Regression

We first trained a classical baseline model using:

- TF-IDF vectorization
- Logistic Regression classifier

Initial results showed:

- High recall for positive samples
- Poor recall for negative samples
- Clear class imbalance behavior

To address this, two improvements were tested:

#### 🔹 Experiment 1: N-grams (1,2)
```python
TfidfVectorizer(ngram_range=(1,2))
```

#### 🔹 Experiment 2: Class Weight Balancing
```python
TfidfVectorizer(class_weight="balanced")
```
## 🛠️ Technologies Used

- Python 3.11
- Hugging Face Datasets
- Transformers
- PyTorch
- NumPy
- Scikit-learn
- Pandas
- Matplotlib

#### 2️⃣ Transition to BERT (Context-Based Modeling)
At this point, it became clear that:

This problem cannot be solved using word frequency alone — context matters.

We therefore switched to a transformer-based model.

🧠 Model Details

-Model: dbmdz/bert-base-turkish-uncased
-Task: Sequence Classification
-Epochs: 2 (only epoch 1 analyzed)
-Goal: Behavioral observation rather than performance maximization

#### 📈 Evaluation Results (Original Text)
- eval_loss     : 0.3256
- eval_accuracy : 0.8758
- eval_f1       : 0.9144
- eval_recall   : 0.7278
- epoch         : 1.0

#### 3️⃣ Text Simplification + BERT Evaluation
Without changing any model parameters, a rule-based text simplification step was applied:

- Complex words replaced with simpler equivalents
- Sentence meaning preserved
- No structural distortion

The same trained BERT model was evaluated on the simplified validation set.

#### 📈 Evaluation Results (Simplified Text)
- eval_loss     : 0.3231
- eval_accuracy : 0.8772
- eval_f1       : 0.9160
- eval_recall   : 0.7135
- epoch         : 1.0

#### 🔮 Future Work

- Multiple datasets
- Different simplification strategies
- Longer training schedules
- Generative or neural simplification models

#### 🚀 Usage
'''python 
git clone https://github.com/your-username/repo.git
python baseline.py
python bert.py
'''

#### ✍️ Related Writing
his project is accompanied by a detailed Medium article explaining the motivation, experiments, and insights.

📎 Medium: (link will be added)


