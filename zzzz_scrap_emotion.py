# failed attempt - version 1

# ---------- Part 1 : Import Dataset ----------
import random
from datasets import load_dataset
dataset = load_dataset("google-research-datasets/go_emotions")

# get the necessary data and labels
data = list(zip(dataset['train']['text'], dataset['train']['labels']))
emotion_labels = dataset['train'].features['labels'].feature.names
random.shuffle(data)

# filter the data (sorry for the profanities)
bad_words = ["fuck", "fucking", "fucked", "fucker", "fucks", "shit", "shitty", "shitting", "shited", "bitch", "bitches", "bitched", "damn", "damned", "goddamn", "asshole", "ass", "bastard", "penis", "vagina", "dick", "cock", "pussy", "ballsack", "clitoris", "cum", "ejaculate", "porn", "pornography", "fetish", "erotic", "retard", "cunt", "slut", "whore", "twat", "fag", "faggot", "nigga", "nigger", "negro", "chink", "spic", "dyke"]
safe_data = [(text, labels) for (text, labels) in data if len(labels) > 0 and not any(word in text.lower() for word in bad_words)]

# review data
print(f'Dataset size: {len(safe_data)}')
for (text, labels) in safe_data[:5]:
    print(f"Text: {text}\nLabels: {[emotion_labels[i] for i in labels]}\n")

# ---------- Part 2 : Data Training ----------
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# turn the texts into vectors (using Tf-Idf)
texts = [text for (text, labels) in safe_data]
label = [emotion_labels[labels[0]] for (text, labels) in safe_data]

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words = 'english')
X = vectorizer.fit_transform(texts)

# split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, label,
    test_size = 0.1,
    random_state = 42
)

# train the MLP
model = MLPClassifier(
    hidden_layer_sizes = (128, 64),
    max_iter = 100,
    alpha = 0.001,
    random_state = 42,
    verbose = False
)
print('Training MLP...')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(classification_report(y_test, predictions))