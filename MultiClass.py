import sklearn as sk
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import struct
from sklearn.model_selection import train_test_split, cross_val_score

# For 32 bit it will return 32 and for 64 bit it will return 64
# print(" system architecture")
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier

print(struct.calcsize("P") * 8)

print(" Reading Multi files")
binaryInputData = pd.read_csv('multi/X_train.csv', sep=',', header=None)
binaryOutputData = pd.read_csv('multi/Y_train.csv', sep=',', header=None)
binaryTestData = pd.read_csv('multi/X_test.csv', sep=',', header=None)
# binaryTestData = pd.concat([binaryInputData, binaryOutputData], axis=1)
# binaryTestData = binaryInputData.join(binaryOutputData)
# binaryTestData = binaryInputData.merge(binaryOutputData)

print("\n Input Dataframe shape")
print(binaryInputData.shape)
# print("\n Input Dataframe columns")
# print(binaryInputData.columns.tolist())
# print("\n Output Dataframe missing values")
# print(binaryInputData.apply(lambda x: sum(x.isnull()), axis=0))

print("\n Output Dataframe shape")
print(binaryOutputData.shape)
print(binaryOutputData)
# print("\n Output Dataawframe columns")
# print(binaryOutputData.columns.tolist())

# Multi class encoding
factor = pd.factorize(binaryOutputData[0])
binaryOutputData[0] = factor[0]
definitions = factor[1]
print(binaryOutputData[0].head())
print(definitions)

print("\n Different outs")
print(binaryOutputData[0].unique())
print(binaryOutputData)

x_train, x_test, y_train, y_test = train_test_split(binaryInputData, binaryOutputData, test_size=0.2)
print("\n split shapes")
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# Get most important features using random forest
print("\n Random forest filtering:")
rf = RandomForestClassifier()
# rf = RandomForestRegressor(max_depth=30, n_estimators=5)
print(" fitting")
rf.fit(x_train, y_train)
print(" Features sorted by their score:")
# lukesingham sort the most important feature
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), x_train), reverse=True))

sfm = SelectFromModel(rf, threshold=0.008)
sfm.fit(x_train, y_train)

X_important_train = sfm.transform(x_train)
X_important_test = sfm.transform(x_test)

X_final_test = sfm.transform(binaryTestData)
print(" Important test features:")
print(X_important_test.shape)
print(" Final test features for CSV:")
print(X_important_test.shape)


# Random forest model

print("\n Random forest")
rfModel = RandomForestClassifier()
rfModel = rfModel.fit(X_important_train, y_train)

RFpredicted = pd.DataFrame(rfModel.predict(X_important_test))
RFpred = pd.DataFrame(rfModel.predict_proba(X_important_test))

print(" Training Score")
print(rfModel.score(X_important_train, y_train))
print(" Score")
print(sk.metrics.accuracy_score(y_test, RFpredicted))
print(" Matrix")
print(sk.metrics.confusion_matrix(y_test, RFpredicted))
print(" Report")
print(sk.metrics.classification_report(y_test, RFpredicted))

# print(" Final prediction")
# finalFrame = pd.DataFrame(rfModel.predict(X_final_test))
# print(finalFrame)
# finalFrame.to_csv("multi/prediction.csv")
# finalFrame[0].replace({0: "whitecoat", 1: "moulted pup", 2: "dead pup", 3: "juvenile", 4: "background"}, inplace=True)
# print(" Refactored final data")
# print(finalFrame)
# finalFrame.to_csv("multi/Y_test.csv")
