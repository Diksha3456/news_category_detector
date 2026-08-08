# 📰 News Article Category Detector

An AI/ML-based text classification project that automatically categorizes a news article into **Sports, Politics, or Tech** using Natural Language Processing (NLP).

## 🚀 Features

* Classifies news articles into:

  * 🏏 Sports
  * 🏛️ Politics
  * 💻 Tech
* Uses **TF-IDF** for text feature extraction
* Uses **Multinomial Naive Bayes** for classification
* Displays prediction confidence
* Simple and interactive **Gradio UI**

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF
* Multinomial Naive Bayes
* Gradio
* Joblib

## 📂 Project Structure

```text
news-category-detector/
│
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── model/
    ├── model.pkl
    └── vectorizer.pkl
```

## ⚙️ How It Works

1. News articles are loaded from the **20 Newsgroups** dataset.
2. Selected categories are grouped into three broader categories: Sports, Politics, and Tech.
3. Text is converted into numerical features using **TF-IDF Vectorization**.
4. A **Multinomial Naive Bayes** classifier is trained on the processed text.
5. The trained model and vectorizer are saved using Joblib.
6. The Gradio interface accepts a news article and predicts its category with a confidence score.

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

This will create:

```text
model/model.pkl
model/vectorizer.pkl
```

### 3. Launch the application

```bash
python app.py
```

The Gradio interface will then be available locally.

## 📊 Model

**Algorithm:** Multinomial Naive Bayes

**Feature Extraction:** TF-IDF

The model is trained on selected categories from the 20 Newsgroups dataset.

## 🎯 Example

**Input:**

> The team won the championship after an exciting final match...

**Output:**

```text
Category: Sports
Confidence: XX.X%
```

## 🔮 Future Improvements

* Add more news categories
* Improve model accuracy with additional algorithms
* Add probability scores for all categories
* Deploy the application online
* Add multilingual news classification

## 👨‍💻 Project

Built as a practical NLP and machine learning project using Python and Gradio.
