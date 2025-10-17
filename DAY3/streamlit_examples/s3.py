import streamlit as st

st.slider('select your age:')
st.slider('select your port:',500,600)
st.slider('select your port:',500,600,550)
st.select_slider('select your rate:',['Bad','Good','Excellent'])

st.selectbox('select your model:',['gpt4.0','gpt5.0','lamma','gemma2:2b'])
st.multiselect('select your model:',['gpt4.0','gpt5.0','lamma','gemma2:2b'])
