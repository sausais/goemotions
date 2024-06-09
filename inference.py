import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

print("[I] Loading test data...", flush=True)
test2 = pd.read_csv("test2.tsv", sep='\t', header=None, names=['emotion', 'text'])

# Encoding
print("[I] Encoding labels...", flush=True)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_classes.npy', allow_pickle=True)
test2['emotion'] = label_encoder.transform(test2['emotion'])

print("[I] Loading tokenizer and model...", flush=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('/home/sausais/projects/valodas/MD2/results/checkpoint-12500')
model.eval()

# Data loading in batches so we do not run out of memory
def create_dataloader(texts, labels, batch_size=16):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(labels))
    return DataLoader(dataset, batch_size=batch_size)

test_loader = create_dataloader(test2['text'].tolist(), test2['emotion'].tolist())

print("[I] Running inference...", flush=True)
all_preds = []
with torch.no_grad():
    for batch in test_loader:
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(predictions)

print("[I] Generating classification report...", flush=True)
report = classification_report(test2['emotion'], all_preds, target_names=label_encoder.classes_, output_dict=True)
print("Classification Report:", flush=True)
print(pd.DataFrame(report).transpose(), flush=True)

print("[I] Generating confusion matrix...", flush=True)
conf_matrix = confusion_matrix(test2['emotion'], all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
