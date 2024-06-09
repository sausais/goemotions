import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import numpy as np

# Load data
print("[I] Loading data...", flush=True)
train2 = pd.read_csv("processed_data/train2.tsv", sep='\t', header=None, names=['emotion', 'text'])
test2 = pd.read_csv("processed_data/test2.tsv", sep='\t', header=None, names=['emotion', 'text'])

# Encode 
print("[I] Encoding labels...", flush=True)
label_encoder = LabelEncoder()
train2['emotion'] = label_encoder.fit_transform(train2['emotion'])
test2['emotion'] = label_encoder.transform(test2['emotion'])

# Save the label classes for inference
np.save('label_classes.npy', label_encoder.classes_)

train_dataset = Dataset.from_pandas(train2)
test_dataset = Dataset.from_pandas(test2)

print("[I] Initializing tokenizer...", flush=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

print("[I] Tokenizing data...", flush=True)
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.rename_column("emotion", "labels")
test_dataset = test_dataset.rename_column("emotion", "labels")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

print("[I] Initializing model...", flush=True)

# Load pretrained model and pass class count
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# logs for tensorboard and dir for results
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return classification_report(p.label_ids, preds, target_names=label_encoder.classes_, output_dict=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("[I] Training the model...", flush=True)
trainer.train()

print("[I] Evaluating the model...", flush=True)
metrics = trainer.evaluate()

# Print classification report
print("Classification Report:", flush=True)
print(metrics, flush=True)