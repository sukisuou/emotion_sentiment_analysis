# ---------- Part 4 : Model Testing ----------
import tensorflow as tf
import numpy as np

# load the labels from dataset
import pickle
with open('emotion_dataset.pkl', 'rb') as file:
    data_pipeline = pickle.load(file)
emotion_labels = data_pipeline['emotion_labels']

# load the model
model = tf.saved_model.load('emotion_model')

# create a module for predicting
def predict(texts):
    texts_arr = np.array(texts, dtype = object)
    return model.serve(texts_arr).numpy()

# ----- test inputs -----
texts = []
print("Enter your comment:")
texts.append(input())

# predict
print('\nAnalysing comment...')
predictions = predict(texts)

# show results
threshold = 0.5     # for sigmoid detection
print('\n>>> Prediction Results <<<\n')
for text, probs in zip(texts, predictions):
    print('-' * 70)
    print(text)
    print('-' * 70)
    print('Result:')

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