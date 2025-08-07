import sys
from functools import wraps
from time import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4
COLUMNS = 17


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r args:[%r, %r] took: %2.4f sec" % (f.__name__, args, kw, te - ts))
        return result

    return wrap


MONTHS = {
    "Jan": 0,
    "Feb": 1,
    "Mar": 2,
    "Apr": 3,
    "May": 4,
    "June": 5,
    "Jul": 6,
    "Aug": 7,
    "Sep": 8,
    "Oct": 9,
    "Nov": 10,
    "Dec": 11,
}


def load_data(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    df = pd.read_csv(
        filename,
        dtype={
            "Administrative": np.int32,
            "Administrative_Duration": np.float64,
            "Informational": np.int32,
            "Informational_Duration": np.float64,
            "ProductRelated": np.int32,
            "ProductRelated_Duration": np.float64,
            "BounceRates": np.float64,
            "ExitRates": np.float64,
            "PageValues": np.float64,
            "SpecialDay": np.float64,
            "OperatingSystems": np.int32,
            "Browser": np.int32,
            "Region": np.int32,
            "TrafficType": np.int32,
        },
    )
    df["Month"] = df["Month"].map(lambda r: MONTHS[r])
    df["VisitorType"] = (
        df["VisitorType"]
        .map(lambda r: 1 if r == "Returning_Visitor" else 0)
        .astype(np.int32)
    )
    df["Weekend"] = df["Weekend"].astype(np.int32)
    df["Revenue"] = df["Revenue"].astype(np.int32)

    label = "Revenue"
    evidence = df.drop(columns=label).to_numpy()
    labels = df[label].to_numpy()

    return (evidence, labels)


def train_model(evidence: np.ndarray, labels: np.ndarray) -> KNeighborsClassifier:
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    print("Learning...")
    model = KNeighborsClassifier(n_neighbors=1)

    model.fit(evidence, labels)

    return model


def evaluate(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    labels = np.array(labels)
    predictions = np.array(predictions)

    tp = np.sum((labels == 1) & (predictions == 1))
    tn = np.sum((labels == 0) & (predictions == 0))

    p = np.sum(labels == 1)
    n = np.sum(labels == 0)

    sensitivity = tp / p if p != 0 else 0.0
    specificity = tn / n if n != 0 else 0.0

    return (
        sensitivity,
        specificity,
    )


if __name__ == "__main__":
    main()
