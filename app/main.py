# ---------- Part 4 : Model Testing ----------
import tensorflow as tf
import numpy as np
import os

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
user_input = False
while True:
    os.system('cls' if os.name == 'nt' else 'clear')

    texts = []
    if user_input:
        print("Enter your comment ('x' to exit):")
        text = input()
        if text == "x":
            print("Bye bye~")
            break
        texts.append(text)
    else:
        texts = [
            "honestly this is one of the best guides i've read in a while, thanks for taking the time to write it",
            "what the hell is this update, literally nobody asked for this and somehow everything is worse now",
            "am i the only one who doesn't understand what's going on here?",
            "for anyone wondering, the answer is in the second paragraph of the guide",
            "i hated that movie, but the ending made it worth it",
            "honestly i don't know if this will work, but im hopeful",
            "wow thanks, that solved absolutely none of my problems"
        ]

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

    if user_input:
        print('\nPress enter to continue...', end = "")
        input()
    else:
        break