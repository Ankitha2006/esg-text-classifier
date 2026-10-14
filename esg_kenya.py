"""
ESG classifier on the KenyaESG dataset (3 separate yes/no classifiers)
----------------------------------------------------------------------
For each pillar (Environmental, Social, Governance) we train one model that
answers: "Is this sentence about this pillar? yes (1) / no (0)".

Needed files (upload them to Colab next to this script):
  KenyaESG_train_environmental.xlsx
  KenyaESG_train_social.xlsx
  KenyaESG_train_governance.xlsx

Run:  python esg_kenya.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, f1_score, accuracy_score

FILES = {
    "Environmental": "KenyaESG_train_environmental.xlsx",
    "Social": "KenyaESG_train_social.xlsx",
    "Governance": "KenyaESG_train_governance.xlsx",
}


def load_pillar(path):
    """Read one xlsx file and return (texts, labels).

    The text column is the one called 'text'. The label column is the other
    column that contains only 0/1 values. This avoids hard-coding names.
    """
    df = pd.read_excel(path)
    print(f"  columns found: {list(df.columns)}")

    text_col = next(c for c in df.columns if str(c).strip().lower() == "text")

    label_col = None
    for c in df.columns:
        if c == text_col:
            continue
        values = set(df[c].dropna().unique())
        if values and values.issubset({0, 1, 0.0, 1.0}) and len(values) == 2:
            label_col = c
            break
    if label_col is None:
        raise ValueError(f"No 0/1 label column found in {path}. Columns: {list(df.columns)}")

    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(int)
    print(f"  using text='{text_col}', label='{label_col}', rows={len(df)}")
    return df[text_col], df[label_col]


def run_pillar(name, path):
    print(f"\n=== {name} ===")
    X, y = load_pillar(path)
    print(f"  positive (yes) share: {y.mean():.1%}")

    # Hold out 20% for testing; stratify keeps the yes/no balance equal.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF: turn sentences into numbers (single words and word pairs).
    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    # class_weight='balanced' helps when 'yes' sentences are rarer than 'no'.
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(Xtr, y_train)

    pred = model.predict(Xte)
    print(classification_report(y_test, pred, target_names=["No", "Yes"], zero_division=0))

    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["No", "Yes"], cmap="Blues", ax=ax)
    ax.set_title(f"{name} - Confusion Matrix")
    fig.tight_layout()
    fig.savefig(f"confusion_matrix_{name.lower()}.png", dpi=150)
    plt.close(fig)

    return {
        "Pillar": name,
        "Train rows": len(X_train),
        "Test rows": len(X_test),
        "Accuracy": round(accuracy_score(y_test, pred), 3),
        "F1 (Yes class)": round(f1_score(y_test, pred), 3),
        "Macro F1": round(f1_score(y_test, pred, average="macro"), 3),
    }


def main():
    results = [run_pillar(name, path) for name, path in FILES.items()]
    summary = pd.DataFrame(results)
    print("\n=== SUMMARY (use these numbers on your CV) ===")
    print(summary.to_string(index=False))
    summary.to_csv("results_summary.csv", index=False)


if __name__ == "__main__":
    main()
