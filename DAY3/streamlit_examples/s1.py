import streamlit as st
import pandas as pd
import numpy as np

st.title("Welcome to streamlit learning")
st.write("this is test message")
st.write("data-1")
st.write("data-2")

df1 = pd.DataFrame({'pid':[101,102,103],'pname':['pA','pB','pC']})
st.write(df1)

df2 = pd.DataFrame(np.random.randn(20,3),columns=['A','B','C'])
st.write(df2)