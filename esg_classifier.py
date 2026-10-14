"""
ESG Text Classifier (beginner-friendly)
---------------------------------------
Goal: classify a sentence from a sustainability report as
Environmental, Social or Governance.

Pipeline:
  1. Load data (text + label)
  2. Split into train and test sets
  3. Convert text to numbers with TF-IDF
  4. Train a Logistic Regression model
  5. Evaluate (accuracy, precision, recall, F1, confusion matrix)
  6. Predict labels for new sentences

Install once:  pip install pandas scikit-learn matplotlib
Run:           python esg_classifier.py

IMPORTANT: the sample data below is only a tiny demo so the script runs.
Replace it with a real dataset (see load_data) before putting this project
on your CV. Search Kaggle or Hugging Face for "ESG sentence classification".
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------------
# STEP 1: Load data
# ---------------------------------------------------------------
SAMPLE_DATA = [
    # Environmental
    ("We reduced our greenhouse gas emissions by 20 percent compared to last year.", "Environmental"),
    ("The company has committed to achieving net-zero carbon emissions by 2040.", "Environmental"),
    ("Renewable energy now supplies 60 percent of our electricity needs.", "Environmental"),
    ("Water consumption at our factories fell through new recycling systems.", "Environmental"),
    ("We expanded waste recycling programs and cut landfill waste significantly.", "Environmental"),
    ("Our science-based targets cover scope 1, 2 and 3 emissions.", "Environmental"),
    ("The plant installed solar panels to lower its carbon footprint.", "Environmental"),
    ("We are protecting biodiversity around our mining sites through reforestation.", "Environmental"),
    ("Energy efficiency upgrades reduced power usage across all offices.", "Environmental"),
    ("We shifted our delivery fleet to electric vehicles to cut air pollution.", "Environmental"),
    # Social
    ("We provide safety training to all employees to reduce workplace injuries.", "Social"),
    ("Women now hold 40 percent of leadership roles in the company.", "Social"),
    ("Our community program funded education for over 5,000 children.", "Social"),
    ("The company offers fair wages and health insurance to every worker.", "Social"),
    ("We run diversity and inclusion workshops for all staff.", "Social"),
    ("Employee turnover decreased after we improved parental leave policies.", "Social"),
    ("Suppliers must follow our human rights and child labour standards.", "Social"),
    ("We donated medical supplies to local hospitals in the region.", "Social"),
    ("Mental health support and counselling are available to all employees.", "Social"),
    ("Our volunteering program gave staff paid time to support local charities.", "Social"),
    # Governance
    ("The board of directors includes a majority of independent members.", "Governance"),
    ("We have a strict anti-corruption and anti-bribery policy.", "Governance"),
    ("Executive pay is linked to long-term performance and shareholder returns.", "Governance"),
    ("An independent audit committee reviews our financial reporting.", "Governance"),
    ("Whistleblower channels allow employees to report misconduct anonymously.", "Governance"),
    ("The company publishes a code of ethics that all managers must follow.", "Governance"),
    ("Shareholders can vote on major decisions at the annual general meeting.", "Governance"),
    ("Our risk management framework is reviewed by the board every quarter.", "Governance"),
    ("We disclose all related-party transactions in the annual report.", "Governance"),
    ("The chair and the chief executive roles are held by different people.", "Governance"),
]


def load_data(csv_path=None):
    """Return a DataFrame with columns 'text' and 'label'.

    To use a real dataset, save it as a CSV with those two columns and call
    load_data("my_esg_data.csv") in main().
    """
    if csv_path:
        return pd.read_csv(csv_path)[["text", "label"]].dropna()
    return pd.DataFrame(SAMPLE_DATA, columns=["text", "label"])


def main():
    df = load_data()
    print(f"Loaded {len(df)} sentences")
    print(df["label"].value_counts(), "\n")

    # -----------------------------------------------------------
    # STEP 2: Train/test split (stratify keeps class balance equal)
    # -----------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.3, random_state=42, stratify=df["label"]
    )

    # -----------------------------------------------------------
    # STEP 3: TF-IDF turns each sentence into a vector of numbers.
    # Words that are frequent in one sentence but rare overall get
    # higher weights. stop_words removes words like "the", "and".
    # -----------------------------------------------------------
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)  # learn vocabulary on train only
    X_test_vec = vectorizer.transform(X_test)        # reuse it on test

    # -----------------------------------------------------------
    # STEP 4: Train Logistic Regression
    # -----------------------------------------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    # -----------------------------------------------------------
    # STEP 5: Evaluate
    # -----------------------------------------------------------
    y_pred = model.predict(X_test_vec)
    print("Classification report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = sorted(df["label"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues")
    plt.title("ESG Classifier - Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")

    # -----------------------------------------------------------
    # STEP 6: Predict on new sentences
    # -----------------------------------------------------------
    new_sentences = [
        "We plan to cut our carbon emissions by half by 2030.",
        "All employees receive equal pay for equal work.",
        "The board reviews executive compensation every year.",
    ]
    preds = model.predict(vectorizer.transform(new_sentences))
    print("\nPredictions on new sentences:")
    for sentence, label in zip(new_sentences, preds):
        print(f"  [{label}] {sentence}")


if __name__ == "__main__":
    main()
