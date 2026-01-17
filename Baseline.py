import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, recall_score

data=load_dataset("BayanDuygu/TrGLUE", "sst2")

train_text=data["train"]["sentence"]
train_label=data["train"]["label"]

val_text=data["validation"]["sentence"]
val_label=data["validation"]["label"]

vectorizer=TfidfVectorizer(max_features=5000,ngram_range=(1,2))

x_train=vectorizer.fit_transform(train_text)
x_val=vectorizer.transform(val_text)

model=LogisticRegression(class_weight="balanced")
model.fit(x_train,train_label)

predictions=model.predict(x_val)

print(f"Başarı Oranı:{accuracy_score(val_label,predictions)}")
print("\n Detaylı Rapor:")
print(classification_report(val_label,predictions))

#confusion matrix
cm=confusion_matrix(val_label,predictions)
disp=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])

# ****
disp.plot(cmap="Blues")
plt.title("Confussion Matrix- Baseline ML")
plt.show()

thresholds=np.arange(0.1,0.9,0.05)

for t in thresholds:
    y_pred_thresh = (y_probs >= t).astype(int)
    f1 = f1_score(val_label, y_pred_thresh)
    recall_0 = recall_score(val_label, y_pred_thresh, pos_label=0)
    print(f"Threshold: {t:.2f} | F1: {f1:.3f} | Class 0 Recall: {recall_0:.3f}")




