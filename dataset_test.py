import random
from datasets import load_dataset
dataset = load_dataset("google-research-datasets/go_emotions")

# get the necessary data and labels
data = list(zip(dataset['train']['text'], dataset['train']['labels']))
emotion_labels = dataset['train'].features['labels'].feature.names
random.shuffle(data)

# preview data
print(f'Dataset size: {len(data)}')
for (text, labels) in data[:10]:
    print(f"Text: {text}\nLabels: {[emotion_labels[i] for i in labels]}\n")