# Yanlış olan: from dataset import load_dataset
# Doğru olan:
from datasets import load_dataset 
import pandas as pd

# Veri setini yükle
dataset = load_dataset("BayanDuygu/TrGLUE", "sst2")
#print(dataset)

df= pd.DataFrame(dataset["train"])#pandas dataframeine ceviriyoruz
#print(df.head())

features=dataset["train"].features["label"]#etiket bilgisi aldık
#print(f"sayısal değerler:{features.names}")

print(f"toplam satır sayısı:{len(df)}")#kaç satır var
print(f"Tablo yapısı (satır, sütun): {df.shape}")#satır sütun bilgisi
