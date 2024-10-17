from transformers import TFBertModel
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, DistilBertTokenizer, TFDistilBertModel, RobertaTokenizer
import torch
import re
import seaborn as sb
from os import listdir
import gc
import psutil
import sys


def prepare_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
        input_text,
        max_length=256,
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors='tf'
    )
    return {
        'input_ids': tf.cast(token.input_ids, tf.float64),
        'attention_mask': tf.cast(token.attention_mask, tf.float64)
    }




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
        tf.print("error single ethn: ", error_single_ethn)
        all_errors.append(error_single_ethn)
        all_indices.append(indexes)
    highest_errors_ind = []
    tf.print("nr of ethn considered: ", int(len(all_errors) / 2))
    for val in range(int(len(all_errors) / 2)):
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
    tf.print("error returned: ", error_high_ethn)
    return error_high_ethn

#distilbert = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
#model = tf.keras.models.load_model("../models/distilbert_50_perc_lowest_ethn_10epochs_balanced_standard_loss",  custom_objects={"TFDistilBertModel": TFDistilBertModel})


def sentiment_score(input_text, classes=[1, 2, 3, 4, 5]):
    # process = psutil.Process()
    # print(process.memory_info().rss)
    processed_data = prepare_data(input_text, tokenizer)
    # print(process.memory_info().rss)
    probs = model(processed_data, training=False)[0]
    # print(classes[np.argmax(probs)])
    # probs = model.predict(processed_data)[0]
    # print(process.memory_info().rss)
    # print()
    return classes[np.argmax(probs)]


# ******************************************************************************************************
model_type = "distilbert"       # distilbert, roberta
setting = "unbalanced"            # balanced, unbalanced
loss_type = "standard_loss"     # standard_loss, custom_loss
# ******************************************************************************************************
models = {}

if model_type == 'distilbert':
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    if setting == "balanced":
        if loss_type == "standard_loss":
            models["distilbert_standard_loss_balanced"] = tf.keras.models.load_model("../models/distilbert_50_perc_lowest_ethn_10epochs_balanced_standard_loss", custom_objects={"TFDistilBertModel": TFDistilBertModel})
        else:
            models["distilbert_custom_loss_balanced"] = tf.keras.models.load_model("../models/distilbert_50_perc_lowest_ethn_10epochs_balanced_custom_loss", custom_objects={'custom_loss': custom_loss, "TFDistilBertModel": TFDistilBertModel})
    if setting == "unbalanced":
        if loss_type == "standard_loss":
            models["distilbert_standard_loss_unbalanced"] = tf.keras.models.load_model("../models/distilbert_50_perc_lowest_ethn_10epochs_unbalanced_standard_loss", custom_objects={"TFDistilBertModel": TFDistilBertModel})
        else:
            models["distilbert_custom_loss_unbalanced"] = tf.keras.models.load_model("../models/distilbert_50_perc_lowest_ethn_10epochs_unbalanced_custom_loss", custom_objects={'custom_loss': custom_loss, "TFDistilBertModel": TFDistilBertModel})

if model_type == 'roberta':
#    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
#    tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    if setting == "balanced":
        if loss_type == "standard_loss":
            models["roberta_standard_loss_balanced"] = tf.keras.models.load_model("../models/roberta_50_perc_lowest_ethn_10epochs_balanced_standard_loss")
        else:
            models["roberta_custom_loss_balanced"] = tf.keras.models.load_model("../models/roberta_50_perc_lowest_ethn_10epochs_balanced_custom_loss", custom_objects={'custom_loss': custom_loss})
    if setting == "unbalanced":
        if loss_type == "standard_loss":
            models["roberta_standard_loss_unbalanced"] = tf.keras.models.load_model("../models/roberta_50_perc_lowest_ethn_10epochs_unbalanced_standard_loss")
        else:
            models["roberta_custom_loss_unbalanced"] = tf.keras.models.load_model("../models/roberta_50_perc_lowest_ethn_10epochs_unbalanced_custom_loss", custom_objects={'custom_loss': custom_loss})

# models["custom_loss_unbalanced"] = tf.keras.models.load_model("../models/sentiment_model_custom_loss_equal_stars_only_10epochs", custom_objects={'custom_loss': custom_loss})
# models["standard_loss_unbalanced"] = tf.keras.models.load_model("../models/sentiment_model_standard_loss_equal_stars_only_10epochs")
#models['BILSTM_unbalanced'] = tf.keras.models.load_model("../models/BILSTM_sentiment_model_unbalanced")

comments = pd.read_csv("../output/comments_equal_sample_equal_stars_3000_classified_standard_loss_model.csv")

for model_key in models.keys():
    comments["predicted_stars_"+model_key] = np.nan
    model = models[model_key]
    tqdm.pandas()
    comments["predicted_stars_"+model_key] = comments['comment'].progress_apply(sentiment_score)

    # iters = 4
    # for i in range(iters):
    #     comments_partition = comments_text[i*len(comments_text)/4, (i+1)*len(comments_text)/4]
    #     preds_partition = comments_partition.apply(sentiment_score)
    #     preds.extend(preds_partition)

    comments.to_csv("../output/comments_equal_sample_equal_stars_3000_classified_" + model_type + "_" +  loss_type + "_" + setting + ".csv", index=False)