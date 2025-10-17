import streamlit as st
import numpy as np
import pandas as pd

st.title('Line Chart Example')

data = pd.DataFrame(np.random.randn(20,3),columns=['A','B','C'])

st.line_chart(data)