import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

keys = [
    #"distilbert_custom_loss_balanced"
    #"distilbert_standard_loss_balanced"
    #,
    #"distilbert_standard",
    #"distilbert_custom_loss_unbalanced"
    "distilbert_standard_loss_unbalanced"
    #
    #"roberta_custom_loss_balanced"
    #,
    #"roberta_standard_loss_balanced"
]

for model_key in keys:
    print(model_key)
    comments = pd.read_csv("../output/comments_equal_sample_equal_stars_3000_classified_" + model_key + ".csv")

    y_true = comments["stars"]
    y_pred = comments["predicted_stars_" + model_key]

    print("Precision")
    print(precision_score(y_true, y_pred, average='macro'))
    print(precision_score(y_true, y_pred, average='micro'))
    print(precision_score(y_true, y_pred, average='weighted'))

    print("Recall")
    print(recall_score(y_true, y_pred, average='macro'))
    print(recall_score(y_true, y_pred, average='micro'))
    print(recall_score(y_true, y_pred, average='weighted'))

    print("F1")
    print(f1_score(y_true, y_pred, average='macro'))
    print(f1_score(y_true, y_pred, average='micro'))
    print(f1_score(y_true, y_pred, average='weighted'))

    print()
    print(classification_report(y_true, y_pred))