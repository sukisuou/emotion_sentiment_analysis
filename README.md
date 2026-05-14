# NLP PROJECT : Emotion Sentiment Analysis

### Group Members: Adam Idlan, Adam Hakimi, Adam Iskandar

## Directory:

1. `emotion_hf.py` : data processing, regex filtering and pipelining for training (may contain profanities)
2. `emotion_train.py` : MLP model training with Tensorflow, Keras and ScikitLearn, with TF-IDF text vectorization
3. `app/main.py` : main code for deployment
4. `app/emotion_dataset.pkl` : cleaned up dataset for training
5. `data_dist.txt` and `training_log.txt` : data insight and training history
6. `zzzz_scrap_emotion.py` : failed attempt with scikit (kept in for proof of trial and error)

## Environment:

To run the project, required packages need to be installed on the system. Otherwise, a virtual environment can be created and used to install packages:

1. Create a virtual environment: `python -m venv .venv`
2. Activate the environment: <br>
   - Linux: `source .venv/bin/activate`<br>

   - Windows: `.venv\Scripts\activate`


3. Install the required package: `pip install packageName`
