import random
from datasets import load_dataset
dataset = load_dataset("google-research-datasets/go_emotions")
random.seed(42)

# get the necessary data and labels
data = list(zip(dataset['train']['text'], dataset['train']['labels']))
emotion_labels = dataset['train'].features['labels'].feature.names

print(dataset)