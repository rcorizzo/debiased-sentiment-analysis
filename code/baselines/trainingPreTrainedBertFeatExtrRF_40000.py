import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KERAS_BACKEND']='tensorflow'
import tensorflow as tf
tf.config.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer
from transformers import TFBertModel

comments = pd.read_csv("../../input/comments_equal_sample_equal_stars_500.csv")
comments = comments.sample(frac=1, random_state=1).reset_index()
labels_train = comments['stars'] # Numeric labels

oHLabel = []
for index, row in comments.iterrows():
    tempval = [0,0,0,0,0]
    tempval[row["stars"]-1]=1
    oHLabel.append(tempval)
comments["oHlabel"] = (oHLabel)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

token = tokenizer.encode_plus(
    comments["comment"].iloc[0],
    max_length=256,
    truncation=True,
    padding='max_length',
    add_special_tokens=True,
    return_tensors='tf'
)

X_input_ids = np.zeros((len(comments),256))
X_attn_masks = np.zeros((len(comments),256))


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

labels = np.zeros((len(comments),5))
labels[np.arange(len(comments)), comments['stars'].values-1] = 1
dataset = tf.data.Dataset.from_tensor_slices((X_input_ids,X_attn_masks,labels))
dataset.take(1)

def SentimentDatasetMapFunctionNoLabel(input_ids, attn_masks, labels):
    return {
        'input_ids': input_ids,
        'attention_mask': attn_masks
    }


dataset = dataset.map(SentimentDatasetMapFunctionNoLabel) # converting to required format for tensorflow dataset
#dataset.take(1)
dataset = dataset.batch(16, drop_remainder=False) # batch size
train_dataset = dataset

bert_model = TFBertModel.from_pretrained('bert-base-cased')

input_ids = tf.keras.layers.Input(shape=(256,), name='input_ids', dtype='int32')
attention_masks = tf.keras.layers.Input(shape=(256,), name='attention_mask', dtype='int32')

bert_embds = bert_model.bert(input_ids,attention_mask=attention_masks)[1]
intermediate_layer = tf.keras.layers.Dense(512,activation='relu', name='intermediate_layer')(bert_embds)
#output_layer = tf.keras.layers.Dense(5, activation='softmax', name = 'output_layer')(intermediate_layer)

model = tf.keras.Model(inputs=[input_ids,attention_masks], outputs = intermediate_layer)
model.summary()

#model = tf.keras.Model(inputs=[input_ids,attention_masks], outputs = output_layer)
#model.summary()

optim = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6)
loss_func = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

#model.compile(optimizer=optim, loss=custom_loss(alpha=alpha), metrics=[acc])
model.compile(optimizer=optim, loss=loss_func, metrics=[acc])  # normal loss

print(np.shape(train_dataset))
print(train_dataset)

# Instead of fine tuning, extract intermediate layer for training reviews
features = model.predict(train_dataset) # I expect this to return a vector of 512 features for each review
print(np.shape(features))
print(features)

print(np.shape(labels_train))
print(labels_train)
# ________________________________
# Train RF on this dataset
# ________________________________
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_model = rf.fit(features,labels_train) # Check if labels are integers or one-hot

# ________________________________
# Load testing dataset and use BERT to extract intermediate features
# ________________________________
comments_testing = pd.read_csv("../../output/comments_equal_sample_equal_stars_3000_classified_standard_loss_model.csv")
#comments_testing = comments_testing.sample(frac=0.005, random_state=1).reset_index()
X_input_ids_testing = np.zeros((len(comments_testing),256))
X_attn_masks_testing = np.zeros((len(comments_testing),256))
X_input_ids_testing, X_attn_masks_testing = generate_training_data(comments_testing, X_input_ids_testing, X_attn_masks_testing, tokenizer)
labels_testing = np.zeros((len(comments_testing),5))
labels_testing[np.arange(len(comments_testing)), comments_testing['stars'].values-1] = 1
dataset_testing = tf.data.Dataset.from_tensor_slices((X_input_ids_testing,X_attn_masks_testing,labels_testing))
dataset_testing.take(1)

dataset_testing = dataset_testing.map(SentimentDatasetMapFunctionNoLabel) # converting to required format for tensorflow dataset
dataset_testing = dataset_testing.batch(16, drop_remainder=False) # batch size

# Apply model.predict again to extract 512 features
features_testing = model.predict(dataset_testing, batch_size=32) # I expect this to return a vector of 512 features for each review
print(np.shape(features_testing))
print(features_testing)

labels_testing = comments_testing["stars"]
print(np.shape(labels_testing))
print(labels_testing)

# ________________________________
# Use RF to test and compute metrics
# ________________________________
rf_preds = rf.predict(features_testing)
print(rf_preds)

# Save Predictions:
comments_testing["predicted_stars_RF_balanced"] = rf_preds
comments_testing.to_csv("../../output/comments_equal_sample_equal_stars_3000_classified_RF_model_balanced_40000.csv", index=False)