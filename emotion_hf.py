# ---------- Part 1 : Import Dataset ----------
import random
from datasets import load_dataset
import os
import dotenv
dotenv.load_dotenv()
dataset = load_dataset("google-research-datasets/go_emotions")

# get the necessary data and labels
data = list(zip(dataset['train']['text'], dataset['train']['labels']))
emotion_labels = dataset['train'].features['labels'].feature.names
random.shuffle(data)

# filter the data using regular expressions
import re
bad_words_string = os.getenv("BAD_WORDS").strip() # .env structure -> BAD_WORDS=word1, word2, ...
bad_words = bad_words_string.split(", ")
profanity_filter = re.compile(r'\b(' + '|'.join(map(re.escape, bad_words)) + r')\b', flags = re.IGNORECASE)

safe_data = []
for text, labels in data:
    if len(labels) == 0:
        continue
    
    if profanity_filter.search(text):
        continue

    safe_data.append((text, labels))

# review data
print(f'Dataset size: {len(safe_data)}')
for (text, labels) in safe_data[:5]:
    print(f"Text: {text}\nLabels: {[emotion_labels[i] for i in labels]}\n")

from collections import Counter
all_labels = [label for text, labels in safe_data for label in labels]
label_counts = Counter(all_labels)
total_labels = len(all_labels)

with open('data_dist.txt', 'w') as file:
    file.write('--- Emotion Distribution ---\n')
    for label_idx, count in label_counts.most_common():
        emotion_name = emotion_labels[label_idx]
        percentage = (count / total_labels) * 100
        file.write(f'{emotion_name}: {count} ({percentage:.2f}%)\n')
with open('data_dist.txt', 'r') as file:
    print(file.read())

# save data
dataset_pipeline = {
    'safe_data': safe_data,
    'emotion_labels': emotion_labels
}
import pickle
with open('app/emotion_dataset.pkl', 'wb') as file:
    pickle.dump(dataset_pipeline, file)