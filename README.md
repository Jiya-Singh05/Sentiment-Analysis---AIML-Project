# 🎭 Sentiment Analyzer

A dual-engine sentiment analysis tool using **VADER** and **TextBlob** — available as both a desktop GUI application and a batch analysis script for IMDB movie reviews.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Output](#output)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview

This project performs sentiment analysis using two popular NLP libraries:

- **VADER** *(Valence Aware Dictionary and sEntiment Reasoner)* — rule-based, optimized for social media and short texts
- **TextBlob** — lexicon-based, great for general-purpose polarity detection

It ships with two components:
1. `app.py` — A sleek **Tkinter GUI** for real-time analysis of custom text
2. `analysis.py` — A **batch analysis script** that evaluates the IMDB dataset and generates accuracy metrics + charts

---

## ✨ Features

- ⚡ Real-time sentiment prediction (Positive / Negative / Neutral)
- 🖥️ Dark-themed desktop GUI with side-by-side VADER and TextBlob results
- 📊 Batch analysis with bar charts and pie charts
- 🧹 Automatic text cleaning (lowercasing, HTML stripping, punctuation removal)
- 🎯 Accuracy evaluation against labeled IMDB data
- 🔄 Threaded analysis to keep the UI responsive

---

## 📁 Project Structure

```
sentiment-analyzer/
│
├── app.py               # Tkinter GUI application
├── analysis.py          # Batch analysis script for IMDB dataset
├── imdb_labelled.csv    # Labeled IMDB movie reviews dataset
│
├── sentiment_chart.png  # Generated bar chart (after running analysis.py)
├── pie_chart.png        # Generated pie chart (after running analysis.py)
│
└── README.md
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/sentiment-analyzer.git
cd sentiment-analyzer
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If you don't have a `requirements.txt`, install manually:
> ```bash
> pip install nltk textblob pandas numpy matplotlib scikit-learn
> ```

### 4. Download NLTK data (auto-handled, but can be done manually)

```python
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')
```

---

## 🖥️ Usage

### Run the GUI App

```bash
python app.py
```

- Type or paste text in the **Quick Input** field or the multi-line **Review** box
- Click **Analyse Sentiment**
- View VADER and TextBlob results side by side
- Use **Clear** to reset

### Run Batch Analysis

```bash
python analysis.py
```

Requires `imdb_labelled.csv` in the same directory. This will:
- Clean and analyze all reviews
- Print accuracy scores and classification reports to the terminal
- Save `sentiment_chart.png` and `pie_chart.png`

---

## ⚙️ How It Works

### Text Cleaning

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)      # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation/numbers
    return text.strip()
```

### VADER Scoring

Returns a **compound score** from `-1.0` (most negative) to `+1.0` (most positive):

| Compound Score | Sentiment |
|----------------|-----------|
| ≥ 0.05         | Positive  |
| ≤ -0.05        | Negative  |
| Between        | Neutral   |

### TextBlob Scoring

Returns a **polarity score** from `-1.0` to `1.0`:

| Polarity | Sentiment |
|----------|-----------|
| > 0      | Positive  |
| < 0      | Negative  |
| = 0      | Neutral   |

---

## 📊 Output

After running `analysis.py`, you'll see:

```
✅ VADER Accuracy:    69.xx%
✅ TextBlob Accuracy: 67.xx%
```

Along with a full classification report and two saved charts:

| Chart | Description |
|-------|-------------|
| `sentiment_chart.png` | Bar chart comparing actual vs. predicted distributions |
| `pie_chart.png`       | Pie chart of VADER and TextBlob sentiment breakdowns |

---

## 📦 Dependencies

| Library       | Purpose                              |
|---------------|--------------------------------------|
| `nltk`        | VADER sentiment analyzer             |
| `textblob`    | TextBlob sentiment analyzer          |
| `pandas`      | Data loading and manipulation        |
| `numpy`       | Numerical operations                 |
| `matplotlib`  | Chart generation                     |
| `scikit-learn`| Accuracy and classification metrics  |
| `tkinter`     | GUI framework (built into Python)    |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

> Built with ❤️ using Python, NLTK, and TextBlob
