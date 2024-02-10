import pandas as pd
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
from lazypredict.Supervised import LazyClassifier


import streamlit as st

st.title("Diabetes Predictor")

gender = st.radio("Select Gender: ", ('Male', 'Female'))
age = st.number_input('Age')
hypertension = st.checkbox("Hypertension ?")
heart_disease = st.checkbox("Heart Disease ?")
bmi = st.number_input('BMI')
hb1ac = st.number_input('HbA1c Level')
b_g = st.number_input('Blood Glucose')

male = 0
female = 0

if(gender == 'Male'):
    male = 1
    female = 0
else:
    female = 1
    male = 0





df = pd.read_csv("C:\\Users\\11110\\OneDrive\\Desktop\\dpd.csv")

df = df.drop('smoking_history',axis='columns')

one_hot = pd.get_dummies(df.gender)

df = pd.concat([df, one_hot], axis="columns")

df = df.drop(["gender","Other"],axis=1)

Y = df.diabetes

X = df.drop('diabetes',axis=1)

train_X,test_X,train_Y,test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)


clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(train_X, test_X, train_Y, test_Y)

print(models)


LR = LogisticRegression(solver='newton-cg').fit(train_X,train_Y)

print(LR.score(test_X,test_Y))

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier()

RF.fit(train_X, train_Y)

print(RF.score(test_X,test_Y))


lr = LR.predict(np.reshape([age,int(hypertension),int(heart_disease),bmi,hb1ac,b_g,male,female],(1,-1)))[0]
rf = RF.predict(np.reshape([[age,int(hypertension),int(heart_disease),bmi,hb1ac,b_g,male,female]],(1,-1)))[0]

st.success(f"{lr} and {rf}")
