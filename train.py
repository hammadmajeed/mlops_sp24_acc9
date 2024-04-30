import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import json
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

df = pd.read_csv("data_processed.csv")

#### Get features ready to model! 
y = df.pop("cons_general").to_numpy()
y[y< 4] = 0
y[y>= 4] = 1

X = df.to_numpy()
X = preprocessing.scale(X) # Is standard
# Impute NaNs

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X)
X = imp.transform(X)


# Logistic Regression model
clf_lr = LogisticRegression()
yhat_lr = cross_val_predict(clf_lr, X, y, cv=5)

acc_lr = np.mean(yhat_lr == y)
tn_lr, fp_lr, fn_lr, tp_lr = confusion_matrix(y, yhat_lr).ravel()
specificity_lr = tn_lr / (tn_lr + fp_lr)
sensitivity_lr = tp_lr / (tp_lr + fn_lr)


# Now print to file
with open("metrics_logistic_regression.json", 'w') as outfile:
    json.dump({
        "accuracy": acc_lr,
        "specificity": specificity_lr,
        "sensitivity": sensitivity_lr
    }, outfile)

# Bar plot by region

sns.set_color_codes("dark")
ax = sns.barplot(x="region", y="pred_accuracy", data=df, palette = "Greens_d")
ax.set(xlabel="Region", ylabel = "Model accuracy")
plt.savefig("by_region.png",dpi=80)
