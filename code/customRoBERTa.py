import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.backend as kb
from transformers import BertTokenizer, AutoModel, AutoTokenizer, RobertaTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForSeq2SeqLM

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# LARGE DATASETS 240K NOT USED AT THE END
#comments = pd.read_csv("../input/comments_equal_stars_train.csv")
#comments = comments.sample(frac=1, random_state=1).reset_index()

# ______________________________________________________________________-
# LOSS TYPE
#loss_type = "custom_loss"
loss_type = "standard_loss"
# ______________________________________________________________________-
# BALANCED
comments = pd.read_csv("../input/comments_equal_sample_equal_stars_500.csv")
setting_str = "balanced"
comments = comments.sample(frac=1)
comments = comments.sort_values(by="stars")
print(comments[0:100])
# ______________________________________________________________________-
# UNBALANCED
#comments = pd.read_csv("../input/comments_unequal_sample_40000.csv")
#setting_str = "unbalanced"
# ______________________________________________________________________-

ethn_dict = {}
i = 0
for ethn in list(comments["ethnicities"].unique()):
    ethn_dict[ethn] = i
    i += 1

oHLabel = []
for index, row in comments.iterrows():
    tempval = [0, 0, 0, 0, 0, 0]
    tempval[row["stars"] - 1] = 1
    tempval[-1] = ethn_dict[row["ethnicities"]]
    oHLabel.append(tempval)
comments["oHlabel"] = (oHLabel)

#tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
#tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

token = tokenizer.encode_plus(
    comments["comment"].iloc[0],
    max_length=256,
    truncation=True,
    padding='max_length',
    add_special_tokens=True,
    return_tensors='tf'
)

X_input_ids = np.zeros((len(comments), 256))
X_attn_masks = np.zeros((len(comments), 256))

X_input_ids.shape


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


X_input_ids, X_attn_masks = generate_training_data(comments, X_input_ids, X_attn_masks, tokenizer)

if loss_type == 'custom_loss':
    print("Custom loss")
    labels_org = comments["oHlabel"].to_numpy()
    print(labels_org)
    labels = np.zeros((len(comments), 6))
    i = 0
    for label in labels_org:
        label = np.array(label)
        labels[i] = label
        i += 1

else:
    labels = np.zeros((len(comments), 5))
    labels[np.arange(len(comments)), comments['stars'].values - 1] = 1


dataset = tf.data.Dataset.from_tensor_slices((X_input_ids, X_attn_masks, labels))

dataset.take(1)


def SentimentDatasetMapFunction(input_ids, attn_masks, labels):
    return {
               'input_ids': input_ids,
               'attention_mask': attn_masks
           }, labels


dataset = dataset.map(SentimentDatasetMapFunction)  # converting to required format for tensorflow dataset

dataset.take(1)

dataset = dataset.shuffle(40000).batch(8, drop_remainder=True)  # batch size, drop any left out tensor

p = 1.0
train_size = int((len(comments) // 8) * p)

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

from transformers import TFBertModel

#bert_model = TFBertModel.from_pretrained('bert-base-cased')
#bert_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
bert_model = TFBertModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

input_ids = tf.keras.layers.Input(shape=(256,), name='input_ids', dtype='int32')
attention_masks = tf.keras.layers.Input(shape=(256,), name='attention_mask', dtype='int32')

bert_embds = bert_model.bert(input_ids, attention_mask=attention_masks)[1]
intermediate_layer = tf.keras.layers.Dense(512, activation='relu', name='intermediate_layer')(bert_embds)
output_layer = tf.keras.layers.Dense(5, activation='softmax', name='output_layer')(intermediate_layer)

model = tf.keras.Model(inputs=[input_ids, attention_masks], outputs=output_layer)

optim = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6)
loss_func = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

def custom_loss(y_true, y_pred):
    # get list of unique ethn:
    unique_ethn = []
    for eth in y_true[:, 5:]:
        if eth not in unique_ethn:
            unique_ethn.append(eth)

    all_errors = []
    all_indices = []

    # per ethn, calculate the average error, save max:
    for ethn in unique_ethn:
        # safe all indexes od current ethn
        indexes = []
        for i in range(len(y_true)):
            if y_true[i][-1] == ethn:
                indexes.append([i])
        # gather only values of current ethn
        y_true_single_ethn = tf.gather_nd(indices=indexes, params=y_true)[:, :5]
        y_pred_single_ethn = tf.gather_nd(indices=indexes, params=y_pred)[:, :5]
        # calculate cross enthropy error of current ethn
        cce = tf.keras.losses.CategoricalCrossentropy()
        error_single_ethn = cce(y_true_single_ethn, y_pred_single_ethn)
        # tf.print("error single ethn: ",error_single_ethn)
        all_errors.append(error_single_ethn)
        all_indices.append(indexes)
    highest_errors_ind = []
    # tf.print("nr of ethn considered: ", int(len(all_errors)/2))
    for val in range(int(len(all_errors) / 2) + 1):
        max_value = max(all_errors)
        index = all_errors.index(max_value)
        # highest_errors_ind.append(all_indices[index])
        highest_errors_ind = highest_errors_ind + all_indices[index]
        all_errors.remove(max_value)
        all_indices.remove(all_indices[index])

    y_true_high_ethn = tf.gather_nd(indices=highest_errors_ind, params=y_true)[:, :5]
    y_pred_high_ethn = tf.gather_nd(indices=highest_errors_ind, params=y_pred)[:, :5]
    # calculate cross enthropy error of 3 ethn
    cce = tf.keras.losses.CategoricalCrossentropy()
    error_high_ethn = cce(y_true_high_ethn, y_pred_high_ethn)
    # tf.print("error returned: ", error_high_ethn)
    return error_high_ethn


# model.compile(optimizer=optim, loss=custom_loss(alpha=alpha), metrics=[acc])
if loss_type == 'custom_loss':
    print('Custom loss')
    model.compile(optimizer=optim, loss=custom_loss, metrics=[acc], run_eagerly=True)
else:
    print('Standard loss')
    loss_func = tf.keras.losses.CategoricalCrossentropy()
    model.compile(optimizer=optim, loss=loss_func, metrics=[acc], run_eagerly=True)

# model.compile(optimizer=optim, loss=loss_func, metrics=[acc])  # normal loss


hist = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=12
)

model.save('roberta_50_perc_lowest_ethn_10epochs_' + setting_str + "_" + loss_type)

