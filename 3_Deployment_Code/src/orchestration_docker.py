import os
from typing import Any, List

import pandas as pd
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab
from joblib import load

import libs_docker.train_docker as CTrain
from libs_docker.preprocess import Preprocess

raw_data_path = "data/"
dest_path = "output/"
train_dataset_filename = "data/car_data.csv"

preprocess = Preprocess(raw_data_path, dest_path, train_dataset_filename)


def read_csv2df(dataset_filename: str) -> pd.DataFrame:
    """Read CSV DataFrame

    Returns:
        pd.DataFrame: Processed pandas DataFrame ready to use.
    """
    df = pd.read_csv(dataset_filename)

    categorical_variables = ["Gender"]
    df_final = pd.get_dummies(df, columns=categorical_variables, drop_first=True)
    df_final = df_final.drop("User ID", axis=1)

    # Data drift
    data_drift_report = Dashboard(tabs=[DataDriftTab()])
    data_drift_report.calculate(df_final[:75], df_final[75:], column_mapping=None)
    data_drift_report.save("./my_report.html")

    return df_final


def cross_validation(df: pd.DataFrame) -> List:
    """Cross-validation for the dataset

    Args:
        df (pd.DataFrame): Input dataset

    Returns:
        List[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: X_train, X_test, y_train, y_test
    """
    return preprocess.cross_validation(df)


def save_train_test_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
):
    """Save in pkl the training and test data.

    Args:
        X_train (pd.DataFrame): X train
        X_test (pd.DataFrame): X test
        y_train (pd.DataFrame): y train
        y_test (pd.DataFrame): y test
    """
    try:
        preprocess.dump_pickle(
            (X_train, y_train), os.path.join(preprocess.dest_path, "train.pkl")
        )
        preprocess.dump_pickle(
            (X_test, y_test), os.path.join(preprocess.dest_path, "test.pkl")
        )
    except:
        preprocess.dump_pickle(
            (X_train, y_train),  "train.pkl"
        )
        preprocess.dump_pickle(
            (X_test, y_test), "test.pkl"
        )

def train_and_log_model(data_path: str) -> None:
    CTrain.train_and_log_model(data_path)


def run_model_save_pred(df: pd.DataFrame):
    loaded_model = load("./model/model.joblib")
    df["Purchased"] = loaded_model.predict(df)
    df.to_csv("output.csv", index=False)


def car_purchase_prediction():
    # Preprocessing
    df = preprocess.read_csv2dataframe()
    X_train, X_test, y_train, y_test = cross_validation(df)
    save_train_test_data(X_train, X_test, y_train, y_test)

    data_path = "./output"
    # Trainning
    train_and_log_model(data_path)

    # Run execution
    path_to_predict = os.getcwd() + "/data/unknown_batch.csv"
    # Preprocessing batch to predict
    df = read_csv2df(path_to_predict)
    run_model_save_pred(df)


if __name__ == "__main__":
    car_purchase_prediction()
    print("Executed batch prediction")
