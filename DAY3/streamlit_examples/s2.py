import streamlit as st
import pandas as pd
import numpy as np

st.title("Welcome to Stremlit")

st.write("Hello..userA")

model = st.text_input("Enter your model name:")

if model:
	st.write(f"Input model name:{model}")

vector = st.slider("select your vector range:",500,900)
if vector:
	st.write(f"selected vector range:{vector}")

d={}
d['pname']=['pA','pB','pC']
d['dept']=['sales','prod','QA']
d['city']=['City-1','City-2','City-3']


st.selectbox('select your data:',d)






