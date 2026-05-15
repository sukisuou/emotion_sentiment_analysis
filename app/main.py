# ---------- Part 3 : Model Testing ----------
import tensorflow as tf
import numpy as np

# load the labels from dataset
import pickle
with open('emotion_dataset.pkl', 'rb') as file:
    data_pipeline = pickle.load(file)
emotion_labels = data_pipeline['emotion_labels']

# load the model
model = tf.saved_model.load('emotion_model')

# create a function for predicting
def predict(texts):
    texts_arr = np.array(texts, dtype = object)
    return model.serve(texts_arr).numpy()

# ----- test inputs -----
texts = [
    "Holy shit, this is the best thing that's happened to me all year.",
    "Wow, amazing job ruining everything again.",
    "I miss her so much it physically hurts sometimes.",
    "I'm proud of you, seriously. You worked so damn hard for this.",
    "This is fine. Everything is totally fine.",
    "I don't know whether to laugh or cry anymore.",
    "Dude, that's actually fucking hilarious.",
    "I'm really nervous about tomorrow but also kinda excited.",
    "I can't believe they forgot my birthday again.",
    "Honestly? That was disgusting and disappointing."
]

# predict
print('Predicting emotions...')
predictions = predict(texts)

# show results
threshold = 0.3     # for sigmoid detection
print('\n>>> Prediction Results <<<\n')
for text, probs in zip(texts, predictions):
    print('-' * 70)
    print(text)
    print('-' * 70)
    print('Detected emotions:')

    # check all the possible emotions
    found_any = False
    for idx, prob in enumerate(probs):
        if prob >= threshold:
            emotion_name = emotion_labels[idx]
            print(f'   - {emotion_name}: {prob * 100:.2f}%')
            found_any = True
    
    # if none is found, pick the highest
    if not found_any:
        highest_idx = np.argmax(probs)
        emotion_name = emotion_labels[highest_idx]
        print(f'   - {emotion_name}: {probs[highest_idx] * 100:.2f}% (Highest, below threshold)')

    print()