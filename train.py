import os
import pickle
import click

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("experiment_001")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)


def run_train(data_path: str):
    

    mlflow.autolog(disable=True)
    
    with mlflow.start_run():

        mlflow.set_tag('developer', 'Maruf')
        mlflow.log_param('train_data_path', os.path.join(data_path, "train.pkl"))
        mlflow.log_param('val_data_path', os.path.join(data_path, "val.pkl"))
        
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric('rmse', rmse)

        #mlflow.log_artifact(local_path='mlruns/models', artifact_path='artifacts')
        mlflow.log_artifact('output/dv.pkl', artifact_path='preprocessor')
        mlflow.sklearn.log_model(rf, artifact_path='artifacts')

if __name__ == '__main__':
    run_train()