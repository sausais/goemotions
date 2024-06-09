import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train2 = pd.read_csv("processed_data/train2_trimmed.tsv", sep='\t', header=None, names=['emotion', 'text', 'tokens', 'normalized_text', 'trimmed_tokens'])
test2 = pd.read_csv("processed_data/test2_trimmed.tsv", sep='\t', header=None, names=['emotion', 'text', 'tokens', 'normalized_text', 'trimmed_tokens'])

# Join back into a single string
train2['trimmed_text'] = train2['trimmed_tokens'].apply(lambda tokens: ' '.join(eval(tokens)))
test2['trimmed_text'] = test2['trimmed_tokens'].apply(lambda tokens: ' '.join(eval(tokens)))

X_train = train2['trimmed_text']
y_train = train2['emotion']
X_test = test2['trimmed_text']
y_test = test2['emotion']

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Encode 
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

#  DataLoader with WeightedRandomSampler
X_train_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

class_sample_counts = np.bincount(y_train_encoded)
weights = 1. / class_sample_counts
samples_weights = weights[y_train_encoded]

sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)

model = MultinomialNB()
for inputs, labels in train_loader:
    model.partial_fit(inputs.numpy(), labels.numpy(), classes=np.unique(y_train_encoded))

# Evaluate
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))

cm = confusion_matrix(y_test_encoded, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()