# ---------- Part 1 : Import Dataset ----------
import random
from datasets import load_dataset
dataset = load_dataset("google-research-datasets/go_emotions")
random.seed(42)

# get the necessary data and labels
data = list(zip(dataset['train']['text'], dataset['train']['labels']))
emotion_labels = dataset['train'].features['labels'].feature.names

# remove data with empty labels
data = [(text, labels) for text, labels in data if len(labels) > 0]

# undersample 'neutral' labels (~28% total data, too skewed)
neutral = []
others = []
neutral_idx = emotion_labels.index('neutral')

for text, labels in data:
    if labels == [neutral_idx]:     # remove pure neutral only
        neutral.append((text, labels))
    else:
        others.append((text, labels))
random.shuffle(neutral)

neutral = neutral[:len(neutral) // 2]  # cut in half
data = neutral + others
random.shuffle(data)

# preview data
print(f'Dataset size: {len(data)}')
for (text, labels) in data[:5]:
    print(f"Text: {text}\nLabels: {[emotion_labels[i] for i in labels]}\n")

# review data distribution
from collections import Counter
all_labels = [label for text, labels in data for label in labels]
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
    'data': data,
    'emotion_labels': emotion_labels
}
import pickle
with open('app/emotion_dataset.pkl', 'wb') as file:
    pickle.dump(dataset_pipeline, file)