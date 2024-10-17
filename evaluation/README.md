## classifyingNames.py
It loads a pre-trained model and generates a file with predictions for each review

###### - Balanced Train: “comments_equal_sample_equal_stars_500.csv” - 40000 reviews (5000 by ethnicity)
###### - Unbalanced Train: “comments_unequal_sample_40000.csv” - Balanced by stars (8000) but not by ethnicity
###### - Testing: “comments_equal_sample_equal_stars_3000_classified_standard_loss_model.csv”

## compute_metrics.py
It loads the prediction file generated by classifyingNames.py and computes metrics