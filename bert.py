from datasets import load_dataset
from transformers import(
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
    #AutoTokenizer
)
import numpy as np
from sklearn.metrics import accuracy_score,f1_score, recall_score


import re

def simplify_text(text):
    replacements = {
        "bireyler": "kişiler",
        "geliştirilmiştir": "yapıldı",
        "amacıyla": "için",
        "kapsamında": "içinde",
        "sağlanmaktadır": "sağlanıyor",
    }

    text = text.lower()
    for k, v in replacements.items():
        text = text.replace(k, v)

    text = re.sub(r"\s+", " ", text).strip()
    return text

dataset=load_dataset("BayanDuygu/TrGLUE", "sst2")

dataset["validation"] = dataset["validation"].map(
    lambda x: {"sentence_simplified": simplify_text(x["sentence"])}
)


#

#tokenizer
tokenizer=BertTokenizerFast.from_pretrained("dbmdz/bert-base-turkish-uncased")

def tokenize_function(batch):
    return tokenizer(batch["sentence"],
                     padding="max_length",
                     truncation=True,
                     max_length=128
    )
def tokenize_simplified(batch):
    return tokenizer(batch["sentence_simplified"],
                     padding="max_length",
                     truncation=True,
                     max_length=128
    )


dataset=dataset.map(tokenize_function, batched=True)

dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

#model
model=BertForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-uncased", num_labels=2)

#metrics
def compute_metrics(eval_pred):
    logits, label=eval_pred
    preds=np.argmax(logits,axis=1)
    return{
        "accuracy":accuracy_score(label,preds),
        "f1":f1_score(label,preds),
        "recall":recall_score(label,preds,pos_label=0)
    }
#training
training_args=TrainingArguments(
    output_dir="./bert_results",
    eval_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    learning_rate=2e-5,
    logging_steps=200,
    save_strategy="no"
)    
trainer=Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_metrics

)

simplified_val = dataset["validation"].map(
    tokenize_simplified,
    batched=True
)

simplified_val.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "label"]
)

trainer.train()
trainer.evaluate(eval_dataset=simplified_val)
trainer.evaluate()