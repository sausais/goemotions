import json
import pandas as pd
import nltk
from collections import Counter
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
import re

# Lejupielādē nepieciešamos NLTK resursus
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Ielādē "ekman_mapping.json" failu
with open("ekman_mapping.json", "r") as f:
    ekman_mapping = json.load(f)

# Ielādē "emotions.txt" failu
with open("emotions.txt", "r") as f:
    emotions = f.read().splitlines()

# Izveido reverse_mapping, lai pārveidotu emocijas uz ekman_emotion
reverse_mapping = {}
for ekman_emotion, emotion_list in ekman_mapping.items():
    for emotion in emotion_list:
        reverse_mapping[emotions.index(emotion)] = ekman_emotion
reverse_mapping[27] = 'neutral'

# Funkcija, lai pārveidotu emocijas indeksus uz tekstuālo reprezentāciju
def convert_tags(file):
    df = pd.read_csv(file, sep='\t', header=None, names=['text', 'emotion_index', 'id'])
    df_expanded = pd.DataFrame(columns=['emotion', 'text'])
    
    for index, row in df.iterrows():
        emotion_indices = row['emotion_index'].split(',')
        for emotion_index in emotion_indices:
            emotion_index = int(emotion_index.strip())
            emotion = reverse_mapping.get(emotion_index, None)
            if emotion is not None:
                df_expanded = df_expanded._append({'emotion': emotion, 'text': row['text']}, ignore_index=True)
    
    return df_expanded

# Pārveido "train.tsv" un "test.tsv" failus
train2 = convert_tags("train.tsv")
test2 = convert_tags("test.tsv")

train2.to_csv("train2.tsv", sep='\t', index=False, header=False)
test2.to_csv("test2.tsv", sep='\t', index=False, header=False)

# Funkcija, lai tokenizētu tekstu
def tokenize_text(text):
    return nltk.word_tokenize(text)

# Tokenizē "train2" un "test2" datus
train2['tokens'] = train2['text'].apply(tokenize_text)
test2['tokens'] = test2['text'].apply(tokenize_text)

# Saglabā tokenizētus datus
train2.to_csv("train2_tokenized.tsv", sep='\t', index=False, header=False)
test2.to_csv("test2_tokenized.tsv", sep='\t', index=False, header=False)

print("Tokenization complete and new files saved successfully.")

# Funkcija, lai analizētu datus un atrastu visbiežāk lietotos vārdus
def analyze_data(df):
    all_tokens = [token for sublist in df['tokens'] for token in sublist]
    frequency = Counter(all_tokens)
    return frequency

frequency = analyze_data(train2)

print("Most common tokens in train2:")
print(frequency.most_common(20))

lemmatizer = WordNetLemmatizer()

# Funkcija, lai iegūtu vārda pozīciju WordNet
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Funkcija, lai normalizētu tekstu
def normalize_text(text):
    # Izņem skaitļus
    text = re.sub(r'\d+', '', text)
    # Tokenizē
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
    return ' '.join(tokens)

# Normalizē "train2" un "test2" datus
train2['normalized_text'] = train2['text'].apply(normalize_text)
test2['normalized_text'] = test2['text'].apply(normalize_text)

# Saglabā normalizētus datus
train2.to_csv("train2_normalized.tsv", sep='\t', index=False, header=False)
test2.to_csv("test2_normalized.tsv", sep='\t', index=False, header=False)

print("Text normalization complete and new files saved successfully.")

stop_words = set(stopwords.words('english'))

# Funkcija, lai iztīrītu leksikonu no pieturvārdiem
def trim_lexicon(tokens):
    return [token for token in tokens if token not in stop_words]

# Iztīra "train2" un "test2" tokenus
train2['trimmed_tokens'] = train2['tokens'].apply(trim_lexicon)
test2['trimmed_tokens'] = test2['tokens'].apply(trim_lexicon)

# Saglabā iztīrītus tokenus
train2.to_csv("train2_trimmed.tsv", sep='\t', index=False, header=False)
test2.to_csv("test2_trimmed.tsv", sep='\t', index=False, header=False)

print("Lexicon trimming complete and new files saved successfully.")
