import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if st.text_input("谁是世界上最帅的猛男") == "孙瑞杰":
    st.write("you are right")
else:
    st.write("you are not right")

st.write("这是数据")
music_data = pd.read_csv("music.csv")
st.write(music_data)
X = music_data.drop(columns=['genre'])
Y = music_data['genre']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
predictions
score = accuracy_score(Y_test, predictions)
score

st.write("我将会给你看一个非常简单的机器学习项目，希望对你的毕设有帮助")
is_clicked = st.button("click me")
st.write("music_data")
st.write(music_data)
st.write("music_data.describe()")
st.write(music_data.describe())
st.write("X=music_data.drop(columns=['genre'])")
st.write(X=music_data.drop(columns=['genre']))
st.write("Y=music_data['genre']")
st.write(Y=music_data['genre'])
st.write("X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2)")
st.write(X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2))
st.write("model=DecisionTreeClassifier()")
st.write(model=DecisionTreeClassifier())
st.write("model.fit(X_train, Y_train)")
st.write(model.fit(X_train, Y_train))
st.write("predictions=model.predict(X_test)")
st.write(predictions=model.predict(X_test))
st.write("predictions")
st.write(predictions)
st.write("score=accuracy_score(Y_test, predictions)")
st.write(score=accuracy_score(Y_test, predictions))
st.write("score")
st.write(score)

chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
st.bar_chart(chart_data)
st.line_chart(chart_data)
