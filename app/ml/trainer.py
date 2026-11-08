import os
import sys
import json
import argparse
import joblib
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from app.ml.features import LinguisticFeatureExtractor

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATASET_PATHS = [
    BASE_DIR / "data" / "truthlens_dataset.csv",
    Path(__file__).resolve().parent / "data" / "truthlens_dataset.csv",
    Path(__file__).resolve().parent.parent / "data" / "truthlens_dataset.csv"
]
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "truthlens_model.joblib"
DEFAULT_METRICS_PATH = Path(__file__).resolve().parent / "model_metrics.json"

def get_builtin_fallback_corpus() -> Tuple[List[str], List[int]]:
    """
    Fallback in-memory corpus of verified Real News and debunked Fake News.
    Preserved for environments where external CSV datasets are absent.
    """
    real_samples = [
        "The World Health Organization confirmed today that global vaccination coverage has reached 72% according to official surveillance data published in Geneva.",
        "NASA's James Webb Space Telescope captured high-resolution infrared imagery of distant galaxies, revealing early cosmic structures formed 13 billion years ago.",
        "The Federal Reserve announced an interest rate adjustment of 25 basis points following the Federal Open Market Committee meeting on economic inflation trends.",
        "The Supreme Court ruled unanimously in favor of clean air standards, upholding regulations issued by the Environmental Protection Agency in Washington.",
        "Ministry of Health officials stated that seasonal influenza cases showed a 14 percent decrease compared to the previous calendar quarter.",
        "Reuters reported that European Central Bank governors met in Frankfurt to review quarterly economic growth targets and fiscal stability indicators.",
        "The Indian Space Research Organisation successfully launched the meteorological satellite into geostationary transfer orbit from Sriharikota.",
        "The Department of Transportation released the comprehensive bridge safety evaluation report following inspections across thirty states.",
        "Associated Press correspondents confirmed the signing of the bilateral trade agreement between diplomatic representatives at the summit in Brussels.",
        "A peer-reviewed clinical trial published in The Lancet demonstrated that the new therapeutic regimen reduced patient recovery duration by four days.",
        "Bank of England governor delivered a keynote address outlining policy measures to mitigate mortgage rate fluctuations in the upcoming fiscal cycle.",
        "The United Nations General Assembly adopted a resolution aimed at preserving international coral reef ecosystems and combating ocean warming.",
        "Researchers at Oxford University published findings documenting genomic variants associated with cardiovascular resilience in over 50,000 participants.",
        "The Bureau of Labor Statistics released employment figures indicating 210,000 nonfarm payroll jobs were added during the preceding thirty-day period.",
        "The Department of Energy announced a $300 million investment in grid modernization and renewable energy storage infrastructure across regional hubs.",
        "Police authorities in London confirmed that three suspects were questioned regarding commercial fraud investigations dating back to November 2024.",
        "A joint statement issued by the Centers for Disease Control and Prevention emphasized routine pediatric health screenings ahead of the academic year.",
        "The international aviation authority concluded its preliminary safety inquiry into engine anomalies reported on scheduled commercial airliners.",
        "Agricultural scientists reported that drought-tolerant hybrid wheat cultivars demonstrated a 15% yield improvement in semi-arid field evaluations.",
        "The International Monetary Fund projected global economic expansion to stabilize at 3.2 percent for the upcoming fiscal calendar."
    ]

    fake_samples = [
        "URGENT ALERT: Drinking boiled garlic water with lemon 100% destroys all viral infections overnight! Big Pharma doctors are desperately trying to ban this secret cure! SHARE BEFORE DELETED!!!",
        "BREAKING: Secret NASA whistleblower reveals world governments are concealing that Earth is completely hollow and inhabited by aliens! They don't want you to know the truth!",
        "Government secretly injecting tracking microchips inside mandatory street lights to control citizen brainwaves using 5G frequencies! SHARE TO WAKE UP THE WORLD!",
        "SHOCKING DISCOVERY: Eating banana peel soaked in vinegar cures stage 4 cancer in just 48 hours without surgery or chemotherapy! Doctors stunned!",
        "WARNING TO ALL CITIZENS: The UN is issuing a new mandatory digital currency that will automatically delete your bank balance if you criticize politicians online!",
        "VIRAL HOAX: Famous Hollywood actor died today in tragic underwater submarine accident! Click here to see the horrifying footage that censors tried to hide!",
        "URGENT MESSAGE FROM DOCTOR: Placing sliced red onions under your bed attracts all household toxins and purifies blood cells while sleeping! Share with all family groups!",
        "SHOCKING PROOF: Drinking pure distilled kerosene drops cures diabetes forever! Hospital cartels hid this ancient remedy to keep selling insulin!",
        "LEAKED DOCUMENT: Scientists secretly genetically engineered mosquitoes to spread mind-control vaccines through tap water across major capital cities!",
        "BREAKING: WhatsApp will start charging 50 dollars per month unless you forward this exact message to at least 25 active contacts immediately! Mark Zuckerberg confirmed!",
        "YOU WON'T BELIEVE THIS: Free government lottery distribution of 10,000 dollars to everyone who clicks this link and forwards their national ID number!",
        "EXPOSED: Secret underground weather machine located in Antarctica was turned on to cause earthquakes and control the world food supply! Spread the truth!",
        "MIRACLE CURE: Rubbing toothpaste on electrical sockets doubles your solar energy output! Power companies are furiously trying to remove this viral video!",
        "ALERT: If you receive a phone call starting with +99, answering it immediately clones your SIM card and transfers your entire bank savings to overseas hackers!",
        "UNBELIEVABLE: Major soft drink company admits their formula contains liquefied microchips to monitor customer purchasing decisions via satellite!",
        "CONFIRMED: Government passes law mandating that all pet dogs must be registered for carbon emissions taxes by next Monday or face confiscation!",
        "BREAKING SCANDAL: Astronauts on the Space Station accidentally dropped the Earth's gravity switch and caused worldwide tidal waves! Censors panicking!",
        "SHOCKING: Putting aluminum foil around your smartphone completely blocks government spying and recharges your battery using ambient magnetic waves!",
        "ALERT TO ALL PARENTS: Toxic plastic eggs filled with poisonous gel are being sold in local grocery markets to hypnotize schoolchildren! Check your pantry now!",
        "SECRET CURE: One teaspoon of baking soda mixed with black pepper cures all cardiac blockages in ten minutes! Cardiologists are terrified you will learn this!"
    ]

    X = real_samples + fake_samples
    y = [0] * len(real_samples) + [1] * len(fake_samples)
    return X, y

def load_dataset(dataset_path: Path = None) -> Tuple[List[str], List[int], str]:
    """
    Loads dataset from an external CSV file if present, else falls back to in-memory corpus.
    Returns: (texts, labels, source_description)
    """
    target_path = None
    if dataset_path and Path(dataset_path).exists():
        target_path = Path(dataset_path)
    else:
        for p in DEFAULT_DATASET_PATHS:
            if p.exists():
                target_path = p
                break

    if target_path:
        try:
            df = pd.read_csv(target_path)
            # Find appropriate text and target columns
            text_col = "text" if "text" in df.columns else ("content" if "content" in df.columns else df.columns[1])
            target_col = "target" if "target" in df.columns else ("label" if "label" in df.columns else df.columns[-1])

            texts = df[text_col].astype(str).tolist()
            raw_targets = df[target_col].tolist()

            labels = []
            for t in raw_targets:
                if str(t).upper() in ["FAKE", "1", "TRUE"]:
                    labels.append(1)
                else:
                    labels.append(0)

            source_desc = f"CSV dataset from '{target_path.name}' ({len(texts)} samples: {labels.count(0)} REAL, {labels.count(1)} FAKE)"
            return texts, labels, source_desc
        except Exception as e:
            print(f"Warning: Failed loading CSV '{target_path}' ({e}). Utilizing fallback corpus.")

    X, y = get_builtin_fallback_corpus()
    source_desc = f"Built-in benchmark corpus ({len(X)} samples: {y.count(0)} REAL, {y.count(1)} FAKE)"
    return X, y, source_desc

def build_training_dataset():
    """Backward-compatible helper function."""
    X, y, _ = load_dataset()
    return X, y

def build_feature_pipeline() -> FeatureUnion:
    """Builds the multi-modal hybrid NLP & stylometric feature extractor."""
    return FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True
        )),
        ("char_tfidf", TfidfVectorizer(
            ngram_range=(3, 5),
            analyzer="char",
            max_features=4000,
            sublinear_tf=True
        )),
        ("linguistics", LinguisticFeatureExtractor())
    ])

def train_and_save_model(model_path=None, dataset_path=None, test_size=0.20, random_state=42) -> Pipeline:
    """
    Trains the TruthLens hybrid NLP & Stylometric fake news classifier
    using a stratified train/test split, calibrated probabilities, and full metric evaluation.
    """
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    else:
        model_path = Path(model_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    X, y, source_desc = load_dataset(dataset_path)
    print("=" * 70)
    print("TruthLens Machine Learning Pipeline Training")
    print("=" * 70)
    print(f"Dataset Source : {source_desc}")
    print(f"Total Samples  : {len(X)} (REAL: {y.count(0)}, FAKE: {y.count(1)})")

    # 2. Stratified train/test split
    if len(X) >= 30:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
    else:
        # Fallback for ultra-small datasets
        X_train, X_test, y_train, y_test = X, X, y, y

    print(f"Train Split    : {len(X_train)} samples")
    print(f"Test Split     : {len(X_test)} samples (Unseen evaluation holdout)")

    # 3. Assemble scikit-learn Pipeline
    features = build_feature_pipeline()
    base_clf = LogisticRegression(C=2.5, max_iter=1000, solver="liblinear", random_state=random_state)
    calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv=3)

    pipeline = Pipeline([
        ("features", features),
        ("classifier", calibrated_clf)
    ])

    # 4. Train pipeline on training split
    pipeline.fit(X_train, y_train)

    # 5. Evaluate on train split
    train_preds = pipeline.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)

    # 6. Evaluate on unseen test holdout split
    test_preds = pipeline.predict(X_test)
    test_probs = pipeline.predict_proba(X_test)

    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds, average="binary", zero_division=0)
    test_rec = recall_score(y_test, test_preds, average="binary", zero_division=0)
    test_f1 = f1_score(y_test, test_preds, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, test_preds)

    print("-" * 70)
    print(f"Training Accuracy : {train_acc * 100:.2f}%")
    print(f"Testing Accuracy  : {test_acc * 100:.2f}%")
    print(f"Testing Precision : {test_prec * 100:.2f}% (Class: FAKE)")
    print(f"Testing Recall    : {test_rec * 100:.2f}% (Class: FAKE)")
    print(f"Testing F1 Score  : {test_f1 * 100:.2f}%")
    print("-" * 70)
    print("Confusion Matrix:")
    print(f"  [[TN={cm[0, 0]:2d}, FP={cm[0, 1]:2d}]")
    print(f"   [FN={cm[1, 0]:2d}, TP={cm[1, 1]:2d}]]")
    print("-" * 70)
    print("Detailed Classification Report:")
    print(classification_report(y_test, test_preds, target_names=["REAL", "FAKE"]))

    # 7. Persist metrics report to JSON
    metrics = {
        "dataset_source": source_desc,
        "total_samples": len(X),
        "real_count": y.count(0),
        "fake_count": y.count(1),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_accuracy": round(float(train_acc) * 100, 2),
        "test_accuracy": round(float(test_acc) * 100, 2),
        "test_precision": round(float(test_prec) * 100, 2),
        "test_recall": round(float(test_rec) * 100, 2),
        "test_f1_score": round(float(test_f1) * 100, 2),
        "confusion_matrix": {
            "true_negatives_real": int(cm[0, 0]),
            "false_positives_fake": int(cm[0, 1]),
            "false_negatives_real": int(cm[1, 0]),
            "true_positives_fake": int(cm[1, 1])
        },
        "algorithm": "CalibratedClassifierCV (Sigmoid, 3-fold) with LogisticRegression (C=2.5, liblinear)",
        "feature_extractors": [
            "Word TF-IDF Vectorizer (ngram=(1, 2), max_features=5000, sublinear_tf=True)",
            "Character TF-IDF Vectorizer (ngram=(3, 5), max_features=4000, sublinear_tf=True)",
            "LinguisticFeatureExtractor (8 handcrafted stylometric & attribution markers)"
        ]
    }

    with open(DEFAULT_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics report exported to {DEFAULT_METRICS_PATH}")

    # 8. Persist trained pipeline estimator
    joblib.dump(pipeline, model_path)
    print(f"Trained model saved to {model_path}")
    print("=" * 70)

    return pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TruthLens Fake News Classifier")
    parser.add_argument("--dataset", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--model-out", type=str, default=None, help="Output path for joblib model")
    args = parser.parse_args()

    train_and_save_model(model_path=args.model_out, dataset_path=args.dataset)
