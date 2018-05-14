from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.externals import joblib
import json
import os
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score

PICKLE_FILE = "qa.pkl"

if not os.path.isfile(PICKLE_FILE):
    with open("ml_train.json") as f:
        results = json.load(f)

    features = [(ft1, ft2) for ft1, ft2, _ in results]
    labels = [label for _, _, label in results]

    logreg = LogisticRegression()
    rfe = RFE(logreg)
    rfe = rfe.fit(features, labels)

    joblib.dump(rfe, PICKLE_FILE)

else:
    rfe = joblib.load(PICKLE_FILE)

with open("ml_dev.json") as f:
    results = json.load(f)

dev_feat = [(ft1, ft2) for ft1, ft2, _ in results]
dev_labels = [label for _, _, label in results]

predictions = rfe.predict(dev_feat)
import pdb; pdb.set_trace()
print("Accuracy score", accuracy_score(predictions, dev_labels))
print("Roc Auc", roc_auc_score(predictions, dev_labels))
print("F1 Score", f1_score(predictions, dev_labels))
