import streamlit as st
import pandas as pd

#st.image('C:\\Users\\karth\\OneDrive\\Pictures\\test1.png')
#st.audio('')
#st.video('')

file = st.file_uploader('select your input file:')

if file:
	df = pd.read_csv(file)
	st.write(df)

st.checkbox("yes")
st.button("Click me")
st.radio("Select your interface:",['eth0','eth1','eth2'])
st.date_input('Travel date:')