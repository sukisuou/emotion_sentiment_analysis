# ---------- Part 3 : Model Evaluation ----------
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, multilabel_confusion_matrix, f1_score, precision_score, recall_score, accuracy_score

# load evaluation data
from datasets import load_dataset
dataset = load_dataset("google-research-datasets/go_emotions")

val_data = list(zip(dataset['validation']['text'], dataset['validation']['labels']))
emotion_labels = dataset['validation'].features['labels'].feature.names

# load the labels and groups from dataset
import pickle
with open('app/emotion_dataset.pkl', 'rb') as file:
    data_pipeline = pickle.load(file)
main_labels = data_pipeline['emotion_labels']
emotion_groups = data_pipeline['emotion_groups']

# label grouping
grouped_val_data = []
for text, labels in val_data:
    grouped_labels = set()
    for label_idx in labels:
        emotion_name = emotion_labels[label_idx]

        for group_idx, emotion_group in enumerate(emotion_groups):
            if emotion_name in emotion_group:
                grouped_labels.add(group_idx)
                break
    
    grouped_val_data.append((text, sorted(list(grouped_labels))))

# load the model
model = tf.saved_model.load('app/emotion_model')

# create a module for predicting
def predict(texts):
    texts_arr = np.array(texts, dtype = object)
    return model.serve(texts_arr).numpy()

# prepare validation X and y
X_val = [text for text, labels in grouped_val_data]
raw_y_val = [labels for text, labels in grouped_val_data]

mlb = MultiLabelBinarizer(classes=list(range(len(main_labels))))
y_true = mlb.fit_transform(raw_y_val)

X_val = np.array(X_val, dtype = object)
y_true = np.array(y_true, dtype = np.int32)

# predict
y_prob = predict(X_val)

# threshold sigmoid outputs
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

with open('validation_log.txt', 'w') as file:   # save in a file
    # evaluate
    file.write('--- Evaluation Scores ---\n')
    file.write(f'Accuracy:  {accuracy_score(y_true, y_pred):.4f}\n')
    file.write(f'Precision: {precision_score(y_true, y_pred, average = "micro", zero_division = 0):.4f}\n')
    file.write(f'Recall:    {recall_score(y_true, y_pred, average = "micro", zero_division = 0):.4f}\n')
    file.write(f'F1 Score:  {f1_score(y_true, y_pred, average = "micro", zero_division = 0):.4f}\n')

    # classification report
    file.write('\n--- Classification Report ---\n')
    file.write(classification_report(
        y_true, y_pred,
        target_names = main_labels,
        zero_division = 0
    ))

    # confusion matrix
    file.write('\n--- Multilabel Confusion Matrix ---\n')
    conf_matrices = multilabel_confusion_matrix(y_true, y_pred)

    for label, matrix in zip(main_labels, conf_matrices):
        file.write(f'\n{label}\n')
        file.write(str(matrix) + '\n')

with open('validation_log.txt', 'r') as file:     # print to terminal
    print(file.read())