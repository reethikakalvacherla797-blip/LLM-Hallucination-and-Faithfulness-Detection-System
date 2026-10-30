"""Evaluate the NLI detector on a small starter dataset.

This file is for testing the workflow only. Replace/extend the small examples
with a properly labeled benchmark such as a suitable HaluEval subset before
reporting meaningful performance numbers.
"""
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from detector import detect_hallucination

DATA = [
    {
        "context": "The Sun is a star.",
        "answer": "The Sun is a star.",
        "actual_label": "SUPPORTED",
    },
    {
        "context": "The Sun is a star.",
        "answer": "The Sun is a planet.",
        "actual_label": "CONTRADICTED",
    },
    {
        "context": "The company was founded in 2010.",
        "answer": "The company has 500 employees.",
        "actual_label": "INSUFFICIENT EVIDENCE",
    },
    {
        "context": "Water freezes at 0 degrees Celsius under standard conditions.",
        "answer": "Water freezes at 0 degrees Celsius under standard conditions.",
        "actual_label": "SUPPORTED",
    },
    {
        "context": "The Earth has one natural moon.",
        "answer": "The Earth has two natural moons.",
        "actual_label": "CONTRADICTED",
    },
    {
        "context": "Python is a programming language.",
        "answer": "Python was created in 1991.",
        "actual_label": "INSUFFICIENT EVIDENCE",
    },
]


def main():
    df = pd.DataFrame(DATA)
    predictions = []

    for index, row in df.iterrows():
        print(f"Checking example {index + 1}/{len(df)}...")
        result = detect_hallucination(row["context"], row["answer"])
        predictions.append(result["status"])

    df["predicted_label"] = predictions
    df.to_csv("evaluation_results.csv", index=False)
    print("\nEvaluation results:")
    print(df[["actual_label", "predicted_label"]].to_string(index=False))

    labels = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT EVIDENCE"]
    print("\nAccuracy:", round(accuracy_score(df["actual_label"], df["predicted_label"]), 4))
    print("\nClassification report:")
    print(classification_report(
        df["actual_label"],
        df["predicted_label"],
        labels=labels,
        zero_division=0
    ))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(df["actual_label"], df["predicted_label"], labels=labels))
    print("\nSaved results to evaluation_results.csv")
    print("Note: six hand-written examples are only a smoke test, not a benchmark.")


if __name__ == "__main__":
    main()
