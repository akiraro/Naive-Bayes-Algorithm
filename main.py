import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

# Importing valset
val = pd.read_csv("data/train.csv")


# Fill Nan value with mean value
val.fillna(val.mean(), inplace=True)

val["Sex_cleaned"] = np.where(val["Sex"] == "male", 0, 1)
val["Embarked_cleaned"] = np.where(val["Embarked"] == "S", 0,np.where(val["Embarked"] == "C", 1,np.where(val["Embarked"] == "Q", 2, 3)))
val["Age_cleaned"] = np.where(val["Age"] >= 45, 0,np.where(val["Age"] >= 18, 1,np.where(val["Age"] >= 0, 2, 3)))


# Split valset in training and test valsets
X_train, X_test = train_test_split(val, test_size=0.3, random_state=int(time.time()))


# Instantiate the classifier
gnb = GaussianNB()
used_features = [
    "Pclass",
    "Sex_cleaned",
    "Age_cleaned",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_cleaned"
]


#  ----- Original ------
gnb.fit(
    X_train[used_features].values,
    X_train["Survived"]
)
y_pred = gnb.predict(X_test[used_features])

# Print results
print("Performance when train valset is divided into half, one part for training and the other half for test :")
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
    .format(
    X_test.shape[0],
    (X_test["Survived"] != y_pred).sum(),
    100 * (1 - (X_test["Survived"] != y_pred).sum() / X_test.shape[0])
))

# --------------------------------- #



#  --------- Modified ----------- #
# Importing test valset
# test = pd.read_csv("val/test.csv")
#
# # Fill Nan value with mean
# test.fillna(test.mean(), inplace=True)
#
# # Clean up some features
# test["Sex_cleaned"] = np.where(test["Sex"] == "male", 0, 1)
# test["Embarked_cleaned"] = np.where(test["Embarked"] == "S", 0,np.where(test["Embarked"] == "C", 1,np.where(test["Embarked"] == "Q", 2, 3)))
#
# gnb.fit(
#     val[used_features].values,
#     val["Survived"]
# )
# y_pred = gnb.predict(test[used_features])
#
# # Save prediction and export to csv
# result = pd.valFrame()
# result['PassengerId'] = pd.Series(test['PassengerId'])
# result['Prediction'] = pd.Series(y_pred)
# result.to_csv("val/result.csv")
# ------------------------------ #

