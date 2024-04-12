import streamlit as st 

st.markdown("<h1 style='text-align: center; color: black;'>About</h1>", unsafe_allow_html=True)



st.markdown("""
    <p style='text-align: center;'>This dashboard is inspired by a dataset published on Kaggle that includes
            some of the works of the famous philosopher Arthur Schopenhauer.
            The dashboard includes some text analysis techniques,
            such as word frequency and sentiment analysis.</p>
""", unsafe_allow_html=True)

st.markdown("<p style='text-align: left; color: black;'><b>Note:</b></p>", unsafe_allow_html=True)

st.markdown("We've noticed some inaccuracies and insufficient processing in our data. We're actively working on fixing these errors to ensure accurate information.")
st.markdown("<p style='text-align: left; color: black;'><b>resources:</b></p>", unsafe_allow_html=True)

st.markdown("[Kaggle](https://www.kaggle.com/datasets/akouaorsot/schopenhauer-work-corpus)")


