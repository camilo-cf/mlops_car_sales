import argparse
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def preprocess_pipeline(X_train):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    return X_train, scaler


def ModelDecisionTreeClassifier(X_train, y_train, X_test, y_test):
    # Scaler
    X_train, scaler = preprocess_pipeline(X_train)
    X_test = scaler.transform(X_test)

    # Decision Tree
    model_name = "DecisionTreeClassifier"
    dtree = DecisionTreeClassifier(random_state=0)
    dtree = dtree.fit(X_train, y_train)
    y_pred_train = dtree.predict(X_train)
    y_pred_test = dtree.predict(X_test)

    # evaluate model on the train and test sets
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    pipeline = Pipeline([("StandardScaler", scaler), ("model", dtree)])

    return model_name, pipeline, train_accuracy, test_accuracy


def ModelXGBClassifier(X_train, y_train, X_test, y_test):
    # Scaler
    X_train, scaler = preprocess_pipeline(X_train)
    X_test = scaler.transform(X_test)

    # XGBClassifier
    model_name = "XGBClassifier"
    xgb_model = XGBClassifier(random_state=0)
    xgb_model.fit(X_train, y_train)
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)

    # evaluate model on the train and test sets
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    pipeline = Pipeline([("StandardScaler", scaler), ("model", xgb_model)])

    return model_name, pipeline, train_accuracy, test_accuracy


def ModelLogisticRegression(X_train, y_train, X_test, y_test):
    # Scaler
    X_train, scaler = preprocess_pipeline(X_train)
    X_test = scaler.transform(X_test)

    # Decision Tree
    model_name = "LogisticRegression"
    logistic_classifier = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred_train = logistic_classifier.predict(X_train)
    y_pred_test = logistic_classifier.predict(X_test)

    # evaluate model on the train and test sets
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)

    pipeline = Pipeline([("StandardScaler", scaler), ("model", logistic_classifier)])

    return model_name, pipeline, train_accuracy, test_accuracy


def train_and_log_model(data_path):
    try:
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))
    except:
        X_train, y_train = load_pickle("train.pkl")
        X_test, y_test = load_pickle("test.pkl")
    best_test_accuracy = 0.0

    # DecisionTreeClassifier
    model_name, model, train_accuracy, test_accuracy = ModelDecisionTreeClassifier(
        X_train, y_train, X_test, y_test
    )
    if test_accuracy > best_test_accuracy:
        best_test_accuracy, final_model_name, final_model, final_train_accuracy = (
            test_accuracy,
            model_name,
            model,
            train_accuracy,
        )

    # ModelLogisticRegression
    model_name, model, train_accuracy, test_accuracy = ModelDecisionTreeClassifier(
        X_train, y_train, X_test, y_test
    )
    if test_accuracy > best_test_accuracy:
        best_test_accuracy, final_model_name, final_model, final_train_accuracy = (
            test_accuracy,
            model_name,
            model,
            train_accuracy,
        )

    # ModelXGBClassifier
    model_name, model, train_accuracy, test_accuracy = ModelXGBClassifier(
        X_train, y_train, X_test, y_test
    )
    if test_accuracy > best_test_accuracy:
        best_test_accuracy, final_model_name, final_model, final_train_accuracy = (
            test_accuracy,
            model_name,
            model,
            train_accuracy,
        )

    from joblib import dump

    dump(model, "./model/model.joblib")

    return (
        final_model_name,
        final_model,
        final_train_accuracy,
        best_test_accuracy,
    )


def run(data_path):
    _, _, _, _ = train_and_log_model(data_path)

    from joblib import load

    loaded_pipeline = load("./model/model.joblib")

    df_batch = pd.read_csv("./data/unknown_batch.csv")
    categorical_variables = ["Gender"]
    df_batch_final = pd.get_dummies(
        df_batch, columns=categorical_variables, drop_first=True
    )
    df_batch_final = df_batch_final.drop("User ID", axis=1)
    df_batch["Purchased"] = loaded_pipeline.predict(df_batch_final)
    df_batch.to_csv("output.csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="the location where the processed data was saved.",
    )
    parser.add_argument(
        "--top_n",
        default=5,
        type=int,
        help="the top 'top_n' models will be evaluated to decide which model to promote.",
    )
    args = parser.parse_args()

    run(args.data_path, args.top_n)
