import argparse
import os
import pickle

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

EXPERIMENT_NAME = "car-purchase-best-models"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


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

    with mlflow.start_run(run_name="DecisionTreeClassifier"):
        # DecisionTreeClassifier
        model_name, model, train_accuracy, test_accuracy = ModelDecisionTreeClassifier(
            X_train, y_train, X_test, y_test
        )
        # Save mode accuracy n the train and test sets
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.sklearn.log_model(model, "model")

    with mlflow.start_run(run_name="ModelLogisticRegression"):
        # ModelLogisticRegression
        model_name, model, train_accuracy, test_accuracy = ModelDecisionTreeClassifier(
            X_train, y_train, X_test, y_test
        )
        # Save mode accuracy n the train and test sets
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.sklearn.log_model(model, "model")

    with mlflow.start_run(run_name="ModelXGBClassifier"):
        # ModelXGBClassifier
        model_name, model, train_accuracy, test_accuracy = ModelXGBClassifier(
            X_train, y_train, X_test, y_test
        )
        # Save mode accuracy n the train and test sets
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)

        mlflow.sklearn.log_model(model, "model")


def run(data_path):

    client = MlflowClient()

    train_and_log_model(data_path)

    # select the model with the lowest test accuracy
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=["metrics.accuracy ASC"],
    )[0]

    # register the best model
    print(best_run)
    print(dir(best_run))
    print(type(best_run))
    best_run_id = best_run.to_dictionary()
    print(best_run_id)
    print(best_run_id.keys())
    # ["run_id"]
    best_run_id = best_run.to_dictionary()["info"]["run_id"]
    print(best_run_id)
    best_model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri=best_model_uri, name="best-car-purchase-classifier")


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
