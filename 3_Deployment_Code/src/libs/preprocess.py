import argparse
import os
import pickle
from typing import Any, List

import pandas as pd
from sklearn.model_selection import train_test_split


class Preprocess:
    def __init__(self, raw_data_path: str, dest_path: str, dataset_filename: str):
        """Inits the preprocessing class.

        Args:
            raw_data_path (str): Location where the raw data was saved.
            dest_path (str): Location where the resulting files will be saved.
            dataset_filename (str): Location of the CSV.
        """
        self.dataset_filename = os.getcwd() + "/" + dataset_filename
        self.dest_path = os.getcwd() + "/" + dest_path
        self.raw_data_path = raw_data_path

    def run(self):
        """Running function of the preprocessing class."""
        # Load files
        df = self.read_csv2dataframe()

        # Data preparation
        X_train, X_test, y_train, y_test = self.cross_validation(df)

        # Create dest_path folder unless it already exists
        os.makedirs(self.dest_path, exist_ok=True)

        # Save datasets
        self.dump_pickle((X_train, y_train), os.path.join(self.dest_path, "train.pkl"))
        self.dump_pickle((X_test, y_test), os.path.join(self.dest_path, "test.pkl"))

        self.dump_pickle((X_train, y_train), "train.pkl")
        self.dump_pickle((X_test, y_test), "test.pkl")

    def read_csv2dataframe(self) -> pd.DataFrame:
        """Read CSV dataframe

        Returns:
            pd.DataFrame: Processed pandas DataFrame ready to use.
        """
        df = pd.read_csv(self.dataset_filename)

        categorical_variables = ["Gender"]
        df_final = pd.get_dummies(df, columns=categorical_variables, drop_first=True)
        df_final = df_final.drop("User ID", axis=1)

        return df_final

    @staticmethod
    def cross_validation(df: pd.DataFrame) -> List:
        """Cross-validation for the dataset

        Args:
            df (pd.DataFrame): Input dataset

        Returns:
            List[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: X_train, X_test, y_train, y_test
        """
        X = df.drop("Purchased", axis=1)
        y = df["Purchased"]
        return train_test_split(X, y, test_size=0.3, random_state=420)

    @staticmethod
    def dump_pickle(obj: Any, filename: str):
        """Save an object in a serialized format

        Args:
            obj (Pipeline): A sklearn.pipeline to save.
            filename (str): Name of the file to save the model.

        Returns:
            str: Location of the saved module.
        """
        with open(filename, "wb") as f_out:
            return pickle.dump(obj, f_out)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path", help="the location where the raw data was saved."
    )
    parser.add_argument(
        "--dest_path", help="the location where the resulting files will be saved."
    )
    parser.add_argument("--dataset_name", help="the name of the source dataset.")
    args = parser.parse_args()

    process = Preprocess(args.raw_data_path, args.dest_path, args.dataset_name)
    process.run()
