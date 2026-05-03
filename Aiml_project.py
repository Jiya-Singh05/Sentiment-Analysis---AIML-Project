import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
nltk.download('stopwords')

print(" All libraries loaded successfully!")

df = pd.read_csv('imdb_labelled.csv')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("Sentiment Counts:")
print(df['sentiment'].value_counts())

print("\nLabel distribution:")
print(df['label'].value_counts())

print("\nMissing values:", df.isnull().sum().sum())

import re
import string

nltk.download('punkt')

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.strip()
    return text

df['clean_review'] = df['review'].apply(clean_text)

print("Text cleaned!")
print("\nOriginal:", df['review'][0])
print("\nCleaned: ", df['clean_review'][0])

analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['vader_sentiment'] = df['clean_review'].apply(get_vader_sentiment)

print(" VADER Analysis Done!")
print("\nVADER Predictions:")
print(df['vader_sentiment'].value_counts())
print(df[['review', 'sentiment', 'vader_sentiment']].head(10))

def get_textblob_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return 'positive'
    elif polarity < 0:
        return 'negative'
    else:
        return 'neutral'

df['textblob_sentiment'] = df['clean_review'].apply(get_textblob_sentiment)

print(" TextBlob Analysis Done!")
print("\nTextBlob Predictions:")
print(df['textblob_sentiment'].value_counts())
print(df[['review', 'sentiment', 'vader_sentiment', 'textblob_sentiment']].head(10))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

actual_counts = df['sentiment'].value_counts()
axes[0].bar(actual_counts.index, actual_counts.values,
            color=['green', 'red', 'gray'])
axes[0].set_title('Actual Sentiment Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Sentiment')
axes[0].set_ylabel('Count')

vader_counts = df['vader_sentiment'].value_counts()
axes[1].bar(vader_counts.index, vader_counts.values,
            color=['green', 'red', 'gray'])
axes[1].set_title('VADER Predictions', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Sentiment')
axes[1].set_ylabel('Count')

textblob_counts = df['textblob_sentiment'].value_counts()
axes[2].bar(textblob_counts.index, textblob_counts.values,
            color=['green', 'red', 'gray'])
axes[2].set_title('TextBlob Predictions', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Sentiment')
axes[2].set_ylabel('Count')

plt.tight_layout()
plt.savefig('sentiment_chart.png', dpi=150)
plt.show()
print(" Charts saved!")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

colors = ['#2ecc71', '#e74c3c', '#95a5a6']

vader_counts = df['vader_sentiment'].value_counts()
axes[0].pie(vader_counts.values, labels=vader_counts.index,
            autopct='%1.1f%%', colors=colors, startangle=90)
axes[0].set_title('VADER Sentiment Breakdown', fontsize=13, fontweight='bold')

tb_counts = df['textblob_sentiment'].value_counts()
axes[1].pie(tb_counts.values, labels=tb_counts.index,
            autopct='%1.1f%%', colors=colors, startangle=90)
axes[1].set_title('TextBlob Sentiment Breakdown', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('pie_chart.png', dpi=150)
plt.show()

from sklearn.metrics import accuracy_score, classification_report

df_binary = df[df['sentiment'].isin(['positive', 'negative'])].copy()

vader_acc = accuracy_score(df_binary['sentiment'], df_binary['vader_sentiment'])
tb_acc = accuracy_score(df_binary['sentiment'], df_binary['textblob_sentiment'])

print(f" VADER Accuracy:    {vader_acc*100:.2f}%")
print(f"TextBlob Accuracy: {tb_acc*100:.2f}%")

print("\n--- VADER Classification Report ---")
print(classification_report(df_binary['sentiment'], df_binary['vader_sentiment']))

def analyze_my_text(text):
    cleaned = clean_text(text)

    vader_result = get_vader_sentiment(cleaned)
    vader_scores = analyzer.polarity_scores(cleaned)

    tb_result = get_textblob_sentiment(cleaned)
    tb_polarity = TextBlob(cleaned).sentiment.polarity

    print(f" Your Text: {text}")
    print(f"\n VADER Result:    {vader_result.upper()}")
    print(f"   Compound Score: {vader_scores['compound']}")
    print(f"\n TextBlob Result: {tb_result.upper()}")
    print(f"   Polarity Score: {tb_polarity:.3f}")

analyze_my_text("movie was very very very very god loved it")
