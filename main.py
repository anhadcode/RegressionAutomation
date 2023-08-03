import streamlit as st
import subprocess

st.title("Automated Regression Analysis")

st.write("""
This tool is dedicated for students who are new into Data science and want to understand how regression analysis works
and get a basic understanding of the same. The good part is you do not to code. You just need to some basic operation
which you will understand while interacting with the interface. 
""")
st.write("""
         The user is expected to upload a dataset. Preferred to be a small dataset because the aim is understanding.
         This tool won't support huge datasets. 
         """)
st.subheader("Linear Regression or Estimation")
"""
INSTRUCTIONS:
1. Select a small dataset, and should not have missing values.
2. Include the necessary rows which can be possible features for the regression.
3. Select the target feature which has _continous_ datatype: numeric data like: 5.645, 89.5, 456

"""


st.subheader("Classification")
"""
INSTRUCTIONS
1. Select a small dataset, and should not have missing values.
2. Include the necessary rows which can be possible features for the regression.
3. Select the target feature which has _Classifying_ datatype: numeric data like: "yes" or "no", 1 or 0

"""


