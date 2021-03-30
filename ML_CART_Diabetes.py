############################################
# DATA SET: DIABETES
###########################################
# Pregnancies: Number of times pregnant
# Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# BloodPressure: Diastolic blood pressure (mm Hg)
# SkinThickness: Triceps skin fold thickness (mm)
# Insulin: 2-Hour serum insulin (mu U/ml)
# BMI: Body mass index (weight in kg/(height in m)^2)
# DiabetesPedigreeFunction: Diabetes pedigree function
# Age: Age (years)
# Outcome: Class variable (0 or 1)
################################################

import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from sklearn.tree import export_graphviz, export_text
from skompiler import skompile
from helpers.data_prep import *
from helpers.eda import *
# pip install imblearn
from imblearn.over_sampling import SMOTE

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def load_dia():
    data = pd.read_csv("C:/Users/Asus/PycharmProjects/bootcamp/dataset/diabetes.csv")
    return data


dia = load_dia()
dia.head()
check_data(dia)

dia_c = dia.copy()
y = dia_c["Outcome"]
X = dia_c.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17)

# Feature Engineering

def dia_data_prep(dataframe):
    # na islemi
    zero_columns = [col for col in dataframe.columns if (dataframe[col] == 0).any() and
                    col not in 'Pregnancies']

    dataframe[zero_columns] = dataframe[zero_columns].replace(0, np.NaN)

    # outliar
    replace_with_thresholds(dataframe, 'SkinThickness')

    # missing
    dataframe.loc[(dataframe['Age'] < 40), 'NEW_AGE_CAT'] = 'adult'
    dataframe.loc[(dataframe['Age'] >= 40) & (dataframe['Age'] < 60), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['Age'] >= 60), 'NEW_AGE_CAT'] = 'senior'
    na_cols = [col for col in dataframe.columns if dataframe[col].isnull().any()]
    for col in na_cols:
        dataframe[col] = dataframe[col].fillna(dataframe.groupby(['NEW_AGE_CAT'])[col].transform("median"))

    # FEATURE ENGINEERING
    dataframe['NEW_BMI_ST'] = dataframe['SkinThickness'] * dataframe['BMI']
    dataframe['NEW_AGE_BMI'] = dataframe['Age'] * dataframe['BMI']
    dataframe['NEW_INS_PREG'] = dataframe['Pregnancies'] * dataframe['Insulin']
    dataframe['NEW_GLU_BP_RATE'] = dataframe['Glucose'] / dataframe['BloodPressure']
    dataframe['NEW_GLU_BMI_RATE'] = dataframe['Glucose'] / dataframe['BMI']
    dataframe['NEW_GLU_DSF_RATE'] = dataframe['Glucose'] / dataframe['DiabetesPedigreeFunction']

    drops = ['Pregnancies', 'Age', 'NEW_AGE_CAT', 'SkinThickness',
             'BloodPressure', 'Insulin']
    dataframe = dataframe.drop(drops, axis=1)

    dataframe.columns = [col.upper() for col in dataframe.columns]

    return dataframe


df = dia_data_prep(dia)

X_train = dia_data_prep(X_train)
X_test = dia_data_prep(X_test)

oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(X_train, y_train)
y_smote.value_counts()

################################
# Model Tuning with holdout
################################

cart_model = DecisionTreeClassifier(random_state=17)

cart_params = {'max_depth': range(1, 15),
               "min_samples_split": [2, 3, 4, 5, 6],
               'min_samples_leaf': [1, 2, 3, 5]}

cart_cv = GridSearchCV(cart_model, cart_params, cv=10, n_jobs=-1, verbose=True)
cart_cv.fit(X_smote, y_smote)

cart_cv.best_params_


cart_tuned = DecisionTreeClassifier(**cart_cv.best_params_).fit(X_smote, y_smote)

# train roc auc score
y_pred = cart_tuned.predict(X_train)
y_prob = cart_tuned.predict_proba(X_train)[:, 1]
print(classification_report(y_train, y_pred))
roc_auc_score(y_train, y_prob)

# test basic roc auc score
y_pred = cart_tuned.predict(X_test)
y_prob = cart_tuned.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

# feature importance

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(cart_tuned, X_train)


# correlation graph
fig, ax = plt.subplots(1, 1, figsize=(20, 15))
vmax=0.5
vmin=-0.5
sns.heatmap(dia.corr(), annot=True, fmt='.2f', cmap='Spectral', mask=np.triu(dia.corr()), vmax=vmax, vmin=vmin, ax=ax)
plt.tight_layout()
plt.show()
