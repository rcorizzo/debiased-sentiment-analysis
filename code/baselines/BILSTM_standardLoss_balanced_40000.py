import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Bidirectional, Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import os

comments = pd.read_csv("../../input/comments_equal_sample_equal_stars_500.csv")
# comments
comments = comments.sample(frac=1, random_state=1).reset_index()


oHLabel = []
for index, row in comments.iterrows():
    tempval = [0,0,0,0,0]
    tempval[row["stars"]-1]=1
    oHLabel.append(tempval)
comments["oHlabel"] = (oHLabel)

def generate_training_data(df, ids, masks, tokenizer):
    for i, text in tqdm(enumerate(df['comment'])):
        tokenized_text = tokenizer.encode_plus(
            text,
            max_length=256,
            truncation=True,
            padding='max_length',
            add_special_tokens=True,
            return_tensors='tf'
        )
        ids[i, :] = tokenized_text.input_ids
        masks[i, :] = tokenized_text.attention_mask
    return ids, masks



labels_org = comments["oHlabel"].to_numpy()
labels = np.zeros((len(comments), 5))
i = 0
for label in labels_org:
    label = np.array(label)
    labels[i] = label
    i += 1



def SentimentDatasetMapFunction(input_ids, attn_masks, labels):
    return {
               'input_ids': input_ids,
               'attention_mask': attn_masks
           }, labels



############## FROM ONLINE KAGGLE NOTEBOOK ################
# model.save('sentiment_model_standard_loss_equal_stars_only_10epochs')


#Using keras to tokenize and pad inputs
tokenize = Tokenizer(oov_token="<OOV>")
tokenize.fit_on_texts(comments["comment"])
word_index = tokenize.word_index
train = tokenize.texts_to_sequences(comments["comment"])
data = pad_sequences(train, padding="post")

#Getting length of the padded input
maxlen = data.shape[1]
print(maxlen)
label = pd.DataFrame(comments["oHlabel"].to_list(), columns=['1','2','3','4','5'])
#Example of padded inputs
print(data[0])
vocab_size = len(word_index) + 1
model = Sequential()
model.add(Embedding(vocab_size, 32))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(5, activation="softmax"))

model.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
history = model.fit(data, label, epochs=15)

model.save('BILSTM_sentiment_model_balanced_40000')
