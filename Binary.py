import sklearn as sk
import sns as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import struct
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score

# For 32 bit it will return 32 and for 64 bit it will return 64
# print(" system architecture")
from sklearn.tree import DecisionTreeClassifier

print(struct.calcsize("P") * 8)

print(" Reading Binary files")
binaryInputData = pd.read_csv('binary/X_train.csv', sep=',', header=None)
binaryOutputData = pd.read_csv('binary/Y_train.csv', sep=',', header=None)
binaryTestData = pd.read_csv('binary/X_test.csv', sep=',', header=None)
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
# print(binaryOutputData)
# print("\n Output Dataframe columns")
# print(binaryOutputData.columns.tolist())

# print("\n Total frame")
# totalframe = pd.concat([binaryInputData, binaryOutputData], axis=1)
# print(totalframe)
# # Heatmap
# correlation = totalframe.corr()
# plt.figure(figsize=(10, 10))
# sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
# plt.show()
# # take a 0.5% sample as this is computationally expensive
# df_sample = totalframe.sample(frac=0.001)
# # Pairwise plots
# pplot = sns.pairplot(df_sample, hue=964)



oneHotOutput = pd.get_dummies(binaryOutputData[0], drop_first=True)
binaryOutputData[0] = oneHotOutput['seal']
print("\n Testing Dataframe OHencoding")
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

sfm = SelectFromModel(rf, threshold=0.01)
sfm.fit(x_train, y_train)

X_important_train = sfm.transform(x_train)
X_important_test = sfm.transform(x_test)
X_final_test = sfm.transform(binaryTestData)
print(" Important test features:")
print(X_important_test.shape)
print(" Final test features for CSV:")
print(X_important_test.shape)

# plottingframe = pd.concat([X_important_train, binaryOutputData], axis=1)
# print(plottingframe)
# # lukesingham Plots
# # Heatmap
# correlation = plottingframe.corr()
# plt.figure(figsize=(10, 10))
# sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
# # take a 0.5% sample as this is computationally expensive
# df_sample = plottingframe.sample(frac=0.005)
# # Pairwise plots
# pplot = sns.pairplot(df_sample, hue=9)

#
# # logistic regression model
# print("\n Logistic regression")
# lrModel = LogisticRegression()
# lrModel = lrModel.fit(X_important_train, y_train)
#
# LRpredicted = pd.DataFrame(lrModel.predict(X_important_test))
# LRpred = pd.DataFrame(lrModel.predict_proba(X_important_test))
# print(" Training Score")
# print(lrModel.score(X_important_train, y_train))
# print(" Score")
# print(sk.metrics.accuracy_score(y_test, LRpredicted))
# print(" Matrix")
# print(sk.metrics.confusion_matrix(y_test, LRpredicted))
# print(" Report")
# print(sk.metrics.classification_report(y_test, LRpredicted))
# print(" Cross validation (10)")
# print(cross_val_score(LogisticRegression(), X_important_test, y_test, scoring='precision', cv=10))
#
#
# # Decision tree model
# print("\n Decision tree")
# dtModel = DecisionTreeClassifier(max_depth=5)
# dtModel = dtModel.fit(X_important_train, y_train)
#
# DTpredicted = pd.DataFrame(dtModel.predict(X_important_test))
# DTpred = pd.DataFrame(dtModel.predict_proba(X_important_test))
# print(" Training Score")
# print(dtModel.score(X_important_train, y_train))
# print(" Score")
# print(sk.metrics.accuracy_score(y_test, DTpredicted))
# print(" Matrix")
# print(sk.metrics.confusion_matrix(y_test, DTpredicted))
# print(" Report")
# print(sk.metrics.classification_report(y_test, DTpredicted))
# print(" Cross validation (10)")
# print(cross_val_score(DecisionTreeClassifier(max_depth=5), X_important_test, y_test, scoring='precision', cv=10))


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
print(" Cross validation (10)")
print(cross_val_score(RandomForestClassifier(), X_important_test, y_test, scoring='precision', cv=10))

# print(" Final prediction")
# finalFrame = pd.DataFrame(rfModel.predict(X_final_test))
# print(finalFrame)
# finalFrame.to_csv("binary/prediction.csv")
# finalFrame[0].replace({0: "background", 1: "seal"}, inplace=True)
# print(" Refactored final data")
# print(finalFrame)
# finalFrame.to_csv("binary/Y_test.csv")

