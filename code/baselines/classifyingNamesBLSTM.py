import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Bidirectional, Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os

#tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

models = {}
# models["custom_loss_unbalanced"] = tf.keras.models.load_model("../models/sentiment_model_custom_loss_equal_stars_only_10epochs", custom_objects={'custom_loss': custom_loss})
# models["standard_loss_unbalanced"] = tf.keras.models.load_model("../models/sentiment_model_standard_loss_equal_stars_only_10epochs")
models['BILSTM_balanced'] = tf.keras.models.load_model("BILSTM_sentiment_model_balanced_40000")
comments_testing = pd.read_csv("../../output/comments_equal_sample_equal_stars_3000_classified_standard_loss_model.csv")

# Refit tokenizer on training data
comments_train = pd.read_csv("../../input/comments_equal_sample_equal_stars_500.csv")
# comments
comments_train = comments_train.sample(frac=1, random_state=1).reset_index()

tokenize = Tokenizer(oov_token="<OOV>")
tokenize.fit_on_texts(comments_train["comment"])
word_index = tokenize.word_index

test = tokenize.texts_to_sequences(comments_testing['comment'])
d = pad_sequences(test, padding="post",maxlen = 256)
predictions = models['BILSTM_balanced'].predict(d)

classes=[1, 2, 3, 4, 5]
preds_processed = []
for elem in predictions:
    preds_processed.append(classes[np.argmax(elem)])
# Save Predictions:
comments_testing["predicted_stars_BLSTM_balanced_40000"] = preds_processed
comments_testing.to_csv("../../output/comments_equal_sample_equal_stars_3000_classified_BLSTM_model_balanced_40000.csv", index=False)
