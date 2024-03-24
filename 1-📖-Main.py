import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import nltk
from nltk import word_tokenize
from collections import Counter
from wordcloud import  WordCloud, STOPWORDS, ImageColorGenerator
from textblob import TextBlob
from PIL import Image

# resources 
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/timeline.html


# data 
df = pd.read_csv("data.csv")

# header
st.set_page_config(
    page_title = 'Arthur schopenhauer',
    page_icon = '📜',
    layout = 'wide')

# title
# st.markdown("<h2 style='text-align: center; color: black;font-family: cursive;'>Arthur schopenhauer</h2>", unsafe_allow_html=True)

#img
image = Image.open('img/arthur3.jpg')
col1, col2, col3 = st.columns(3)
col2.image(image, width=300)

# quote 
quote = "The more unintelligent a man is, the less mysterious existence seems to him. - Arthur schopenhauer-"
styled_quote = f'<blockquote style="font-style: italic ;text-align: center; color: black:font-size:20px;;"><b>{quote}</b></blockquote>'
st.markdown(styled_quote, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


# filter
job_filter = st.selectbox('Select a book:', df['book_title'])

#book title 
# st.markdown(f"<h3 style='text-align: center; color: black;font-family: cursive;'>{job_filter}</h2>", unsafe_allow_html=True)
# st.markdown("<br>", unsafe_allow_html=True)


will = df[df['book_title'] == job_filter]['text_clean']
tokens = word_tokenize(will.iloc[0])  # Access the first row's text

# word frequency
freq = Counter(tokens)
sorted_freq = dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
top_25_words = list(sorted_freq.keys())[:25]
top_25_freq = list(sorted_freq.values())[:25]

col1, col2, col3= st.columns(3)


# bar plot 
#st.write(f"Top 25 words in {job_filter}")
fig, ax = plt.subplots(figsize=(8, 7))
sns.barplot(y=top_25_words, x=top_25_freq, ax=ax)
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.title((f"Top 25 words in {job_filter}"))

col1.pyplot(fig)

# Word Cloud
wordcloud = WordCloud(width = 800, height = 560, random_state=1, 
                      background_color='white', colormap='Set2', 
                      collocations=False, stopwords = STOPWORDS)
wordcloud.generate_from_frequencies(sorted_freq)

col2.image(wordcloud.to_image())


# pichart 
def analyze_sentiment(word):
    analysis = TextBlob(word)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'
words = tokens

sentiments = [analyze_sentiment(word) for word in words]

df_sentiment = pd.DataFrame({'Word': words, 'Sentiment': sentiments})

fig, ax = plt.subplots(figsize=(6, 6))
df_sentiment['Sentiment'].value_counts().plot.pie(autopct='%1.1f%%', colors=['pink', 'orange', 'gray'])
ax.set_title((f"Sentiment in  {job_filter}"))

plt.legend(df_sentiment['Sentiment'].value_counts().index, loc='best')
ax.axis('equal')  

col3.pyplot(fig)


# Time line  

# dates = df['publishing_date'].to_list()
# names = df['book_title'].to_list()
# names[-1] = 'Principle of Sufficient Reason'

# font_size = 20

# # Set default font size for all text elements
# plt.rc('font', size=font_size)                   # Controls default text sizes
# plt.rc('axes', labelsize=font_size)              # Axes label font size
# plt.rc('xtick', labelsize=font_size)             # X-axis tick label font size
# plt.rc('ytick', labelsize=font_size)             # Y-axis tick label font size
# plt.rc('legend', fontsize=font_size)             # Legend font size
# plt.rc('figure', titlesize=font_size)


# # Choose some nice levels
# levels = np.tile([-5, 5, -3, 3, -1, 1], int(np.ceil(len(dates) / 6)))[:len(dates)]

# # Create a figure and plot a stem plot with the date
# fig, ax = plt.subplots(figsize=(45, 20))
# ax.set_title("Chronology of Schopenhauer's publications", fontsize=30)

# ax.vlines(dates, 0, levels, color="tab:red")  # The vertical stems.
# ax.plot(dates, np.zeros_like(dates), "-o", color="k", markerfacecolor="w")  # Baseline and markers on it.

# # Annotate lines
# for d, l, r in zip(dates, levels, names):
#     ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l) * 3), textcoords="offset points",
#                 horizontalalignment="right", verticalalignment="bottom" if l > 0 else "top")

# # Format x-axis with 4-month intervals
# plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

# # Remove y-axis and spines
# ax.yaxis.set_visible(False)
# ax.spines[["left", "top", "right"]].set_visible(False)

# ax.margins(y=0.2)

# # Display the Matplotlib figure in Streamlit
# st.pyplot(fig)

