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

# group data together into 3 big labels - positive, negative, ambiguous
positive_emotions = {
    'admiration', 'amusement', 'approval', 'caring', 'desire',
    'excitement', 'gratitude', 'joy', 'love', 'optimism',
    'pride', 'relief'
}
negative_emotions = {
    'anger', 'annoyance', 'disappointment', 'disapproval', 'disgust',
    'embarrassment', 'fear', 'grief', 'nervousness', 'remorse',
    'sadness'
}
ambiguous_emotions = {
    'confusion', 'curiosity', 'realization', 'surprise', 'neutral'
}
grouped_data = []
main_labels = ['positive', 'negative', 'ambiguous']

for text, labels in data:
    grouped_labels = set()
    for label_idx in labels:
        emotion_name = emotion_labels[label_idx]

        if emotion_name in positive_emotions:
            grouped_labels.add(0)
        elif emotion_name in negative_emotions:
            grouped_labels.add(1)
        elif emotion_name in ambiguous_emotions:
            grouped_labels.add(2)
    
    grouped_data.append((text, sorted(list(grouped_labels))))

# review data distribution
from collections import Counter
all_main_labels = [label for text, labels in grouped_data for label in labels]
main_label_counts = Counter(all_main_labels)
total_main_labels = len(all_main_labels)

with open('label_dist.txt', 'w') as file:
    file.write('--- Main Label Distribution ---\n')
    for label_idx, count in main_label_counts.most_common():
        emotion_name = main_labels[label_idx]
        percentage = (count / total_main_labels) * 100
        file.write(f'{emotion_name}: {count} ({percentage:.2f}%)\n')
with open('label_dist.txt', 'r') as file:
    print(file.read())

# save data
dataset_pipeline = {
    'data': grouped_data,
    'emotion_labels': main_labels
}
import pickle
with open('app/emotion_dataset.pkl', 'wb') as file:
    pickle.dump(dataset_pipeline, file)