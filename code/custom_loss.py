
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer
import tensorflow.keras.backend as kb


comments = pd.read_csv("../input/data.csv")
comments = comments.sample(frac=1, random_state=1).reset_index()


ethn_dict = {}
i = 0
for ethn in list(comments["ethnicities"].unique()):
    ethn_dict[ethn] = i
    i += 1

oHLabel = []
for index, row in comments.iterrows():
    tempval = [0,0,0,0,0,0]
    tempval[row["stars"]-1]=1
    tempval[-1]= ethn_dict[row["ethnicities"]]
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



labels_org = comments["oHlabel"].to_numpy()
labels = np.zeros((len(comments),6))
i=0
for label in labels_org:
    label = np.array(label)
    labels[i]=label
    i+=1


dataset = tf.data.Dataset.from_tensor_slices((X_input_ids,X_attn_masks,labels))
dataset.take(1)


def SentimentDatasetMapFunction(input_ids, attn_masks, labels):
    return {
        'input_ids': input_ids,
        'attention_mask': attn_masks
    }, labels


dataset = dataset.map(SentimentDatasetMapFunction) # converting to required format for tensorflow dataset
dataset.take(1)



dataset = dataset.shuffle(10000).batch(16, drop_remainder=True) # batch size, drop any left out tensor

p = 0.9
train_size = int((len(comments)//16)*p)


train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)


from transformers import TFBertModel


bert_model = TFBertModel.from_pretrained('bert-base-cased')

input_ids = tf.keras.layers.Input(shape=(256,), name='input_ids', dtype='int32')
attention_masks = tf.keras.layers.Input(shape=(256,), name='attention_mask', dtype='int32')

bert_embds = bert_model.bert(input_ids,attention_mask=attention_masks)[1]
intermediate_layer = tf.keras.layers.Dense(512,activation='relu', name='intermediate_layer')(bert_embds)
output_layer = tf.keras.layers.Dense(5, activation='softmax', name = 'output_layer')(intermediate_layer)

model = tf.keras.Model(inputs=[input_ids,attention_masks], outputs = output_layer)

optim = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6)
#loss_func = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')


def custom_loss(y_true, y_pred):
    # get list of unique ethn:
    unique_ethn = []
    for eth in y_true[:,5:]:
        if eth not in unique_ethn:
            unique_ethn.append(eth)
    
    max_error = 0
    #per ethn, calculate the average error, save max: 
    for ethn in unique_ethn:
        # safe all indexes od current ethn
        indexes = []
        for i in range(len(y_true)):
            if y_true[i][-1]==ethn:
                indexes.append([i])
        #gather only values of current ethn
        y_true_single_ethn = tf.gather_nd(indices = indexes, params = y_true)[:,:5]
        y_pred_single_ethn = tf.gather_nd(indices = indexes, params = y_pred)[:,:5]
        #calculate mean error of current ethn
        error_single_ethn = y_pred_single_ethn - y_true_single_ethn
        mean_error_single_ethn = tf.reduce_mean(error_single_ethn)
        #save only greatest absolute error
        if (tf.math.abs(mean_error_single_ethn)>max_error):
            # should I return absolute?
            max_error = tf.math.abs(mean_error_single_ethn)
    return max_error


model.compile(optimizer=optim, loss=custom_loss, metrics=[acc], run_eagerly=True) 



hist = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=1
)



model.save('sentiment_model')

