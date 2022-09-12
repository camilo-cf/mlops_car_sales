from typing import Any, List

import os
import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.task_runners import SequentialTaskRunner

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from datetime import date

from libs.preprocess import Preprocess
import libs.train as CTrain

from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab

mlflow.set_tracking_uri("http://127.0.0.1:5000")
raw_data_path = "data/"
dest_path = "output/"
train_dataset_filename = "data/car_data.csv"

preprocess = Preprocess(raw_data_path, dest_path, train_dataset_filename)

@task
def read_csv2df(dataset_filename: str) -> pd.DataFrame:
    """Read CSV DataFrame

    Returns:
        pd.DataFrame: Processed pandas DataFrame ready to use.
    """
    get_run_logger().info("Reading CSV DataFrame")
    df = pd.read_csv(dataset_filename)

    categorical_variables = ["Gender"]
    df_final = pd.get_dummies(df, columns = categorical_variables, drop_first = True)
    df_final = df_final.drop('User ID', axis = 1)

    # Data drift
    get_run_logger().info("Starting the data drift analysis")
    data_drift_report = Dashboard(tabs=[DataDriftTab()])
    data_drift_report.calculate(df_final[:75], df_final[75:], 
        column_mapping = None)
    report_save = os.path.join(os.getcwd(),"my_report.html")
    data_drift_report.save(report_save)
    mlflow.log_artifact(report_save)
    get_run_logger().info("Data drift analysis saved")


    return df_final

def cross_validation(df: pd.DataFrame) -> List:
    """Cross-validation for the dataset

    Args:
        df (pd.DataFrame): Input dataset

    Returns:
        List[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: X_train, X_test, y_train, y_test
    """
    get_run_logger().info("Loading Cross-validation from external library")
    return  preprocess.cross_validation(df)

@task
def save_train_test_data(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame):
    """Save in pkl the training and test data.

    Args:
        X_train (pd.DataFrame): X train
        X_test (pd.DataFrame): X test
        y_train (pd.DataFrame): y train
        y_test (pd.DataFrame): y test
    """
    get_run_logger().info("Saving test and train data in a local pkl")
    preprocess.dump_pickle((X_train, y_train), os.path.join(preprocess.dest_path, "train.pkl"))
    preprocess.dump_pickle((X_test, y_test), os.path.join(preprocess.dest_path, "test.pkl"))

@task
def train_and_log_model(data_path: str) -> None:
    get_run_logger().info("Training models with an external library")
    CTrain.train_and_log_model(data_path)

@task
def register_model(client: MlflowClient) -> None:
    # Select the model with the best test accuracy
    experiment = client.get_experiment_by_name(CTrain.EXPERIMENT_NAME)
    best_run = client.search_runs(
                    experiment_ids=experiment.experiment_id,
                    run_view_type=ViewType.ACTIVE_ONLY,
                    max_results=5,
                    order_by=["metrics.accuracy ASC"]
                )[0]
    
    # Register the best model
    best_run_id = best_run.to_dictionary()["info"]["run_id"]
    best_model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(model_uri=best_model_uri, name="best-car-purchase-classifier")
    get_run_logger().info(f"Registering the best model with URI {best_model_uri}")
    return best_model_uri

@task
def run_model_save_pred(df: pd.DataFrame, best_model_uri: str):
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(best_model_uri)
    get_run_logger().info(f"Predicting the batch data.")
    df["Purchased"] = loaded_model.predict(df)
    get_run_logger().info(f"Saving the output data in output.csv.")
    df.to_csv("output.csv", index=False)
    mlflow.log_artifact("output.csv")


@flow(task_runner=SequentialTaskRunner())
def car_purchase_prediction(date = date.today()):
    # Preprocessing
    df = preprocess.read_csv2dataframe()
    X_train, X_test, y_train, y_test = cross_validation(df)
    save_train_test_data(X_train, X_test, y_train, y_test)

    data_path = "./output"
    # Trainning
    client = MlflowClient()
    train_and_log_model(data_path)
    # Registering model
    best_model_uri = register_model(client)

    # Run execution
    path_to_predict = os.getcwd()+"/data/unknown_batch.csv"
    # Preprocessing batch to predict
    df = read_csv2df(path_to_predict)
    run_model_save_pred(df, best_model_uri)

if __name__ == "__main__":
    car_purchase_prediction()

