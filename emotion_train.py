# ---------- Part 2 : Data Training ----------
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization, TextVectorization
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, F1Score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# import data
import pickle
with open('app/emotion_dataset.pkl', 'rb') as file:
    data_pipeline = pickle.load(file)

data = data_pipeline['data']
emotion_labels = data_pipeline['emotion_labels']

# prepare the data
X = [text for (text, labels) in data]
raw_labels = [labels for (text, labels) in data]
mlb = MultiLabelBinarizer(classes = list(range(len(emotion_labels))))
y = mlb.fit_transform(raw_labels)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = 42
)
X_train = np.array(X_train, dtype = object)
X_test = np.array(X_test, dtype = object)
y_train = np.array(y_train, dtype = np.int32)
y_test = np.array(y_test, dtype = np.int32)

# vectorize text using tf-idf unigram/bigram features
max_tokens = 8000
tfidf_layer = TextVectorization(
    max_tokens = max_tokens,
    output_mode = 'tf_idf',
    ngrams = (1, 2)
)
tfidf_layer.adapt(X_train)

# use class weighting to make the dataset more balanced - boost rare classes weights, keep normal classes weights
label_counts = y_train.sum(axis = 0)
total_samples = y_train.shape[0]
label_counts = np.maximum(label_counts, 1)  # avoid div by 0
multiplier = 1.2    # strength of positive weights

positive_weights = multiplier * total_samples / (len(emotion_labels) * label_counts)
positive_weights = tf.constant(positive_weights, dtype = tf.float32)

# calculate bce with weighted classes
def weighted_binary_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    weights = y_true * positive_weights + (1.0 - y_true)

    return tf.reduce_mean(bce * weights)

# build MLP model
model = tf.keras.Sequential([
    Input(shape = (), dtype = tf.string),

    # vectorized layer
    tfidf_layer,

    # first layer
    Dense(256, name = 'first_layer'),
    LeakyReLU(alpha = 0.01),
    Dropout(0.4),

    # normalize layer for stability
    BatchNormalization(),

    # second layer
    Dense(128, name = 'second_layer'),
    LeakyReLU(alpha = 0.01),
    Dropout(0.2),

    # output layer
    Dense(len(emotion_labels), activation = 'sigmoid', name = 'output')
])
model.compile(
    optimizer = tf.keras.optimizers.AdamW(  # use adamw for weight decay
        learning_rate = 5e-4,
        weight_decay = 1e-4
    ),
    loss = weighted_binary_crossentropy,    # use own function
    metrics = [
        BinaryAccuracy(name = 'accuracy'),
        Precision(name = 'precision'),
        Recall(name = 'recall'),
        F1Score(average = 'micro', name = 'f1_score')
    ]
)

# add early stopping to avoid overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_f1_score',
    mode = 'max',
    patience = 7,
    restore_best_weights = True
)
model.summary()

# train model 
print('Training model...')
history = model.fit(
    X_train, y_train,
    epochs = 25,
    batch_size = 32,
    verbose = False,
    validation_data = (X_test, y_test),
    callbacks = [early_stopping]
)
print('Training done!')

# evaluate model's metrics
history_dict = list(history.history.items())
with open('training_log.txt', 'w') as file:     # save in a file
    file.write(f"Stopped at epoch: {len(history.history['loss'])}\n")
    file.write('\n--- Training Scores ---\n')
    for metrics_name, score in history_dict:
        if not metrics_name.startswith('val_'):
            file.write(f'{metrics_name}: {score[-1]:.4f}\n')
    file.write('\n--- Evaluation Scores ---\n')
    for metrics_name, score in history_dict:
        if metrics_name.startswith('val_'):
            file.write(f'{metrics_name}: {score[-1]:.4f}\n')
with open('training_log.txt', 'r') as file:     # print to terminal
    print(file.read())

# save model
model.export('app/emotion_model')