# 1. Problem definition and Solution Approach

## Problem description

The **Bank DZC** collects customers' personal data like age, annual income, address, phone number, etc.

Currently, **the bank is promoting loans for car acquisition to users more likely to buy a car**.

![Car's Loan](https://img.freepik.com/free-vector/car-finance-concept-illustration_114360-8058.jpg)

## Proposed Solution

The bank identified [this dataset in Kaggle](https://www.kaggle.com/datasets/gabrielsantello/cars-purchase-decision-dataset?resource=download), which contains labeled data about people with variables like Gender, Age, Annual Salary, and if the person Purchased a car or not.

So then, after [analyzing the data (in this notebook)](1_EDA/0_EDA.ipynb), the bank decided to create an ML solution to identify the potential clients to offer them a vehicle credit.

An offline deployment can solve the bank's problem, particularly a **batch solution.**


## Goal
Calculate once a month for all the active users if they are likely to buy a car and provide the results to the commercial area to approach the target users with the right offers.

## Experiment Setup
As I do not count with batch data to apply the model in this hypothetical case, I created a script that creates synthetic data to simulates the process. The synthetic data is created with [this script](2_ML_modeling/0_create_fake_data.ipynb), where the EDA values are taken to create the related dataset. Created users: 1000.

# 2. Cloud Services to be used

The current service can be deployed in Azure and AWS with a **Docker container** or a **Prefect Instance**:

## 2.1 Docker Container
1. In an Ubuntu Server 20.04 LTS or 22.04 LTS virtual machine with Docker installed. The tested VM were:
     - AWS EC2 (in a VM `t4g.large`).
     - Azure Virtual machine (in a `D4pls v5`).

     The execution steps for the VM should follow [this script](./run_docker.sh), that:
     1. Delete the latest output of the algorithm in CSV.
     2. Build the [Dockerfile](Dockerfile).
     3. Run the build [Dockerfile](Dockerfile) with shared local storage.

2. This project was tested in AWS ECR triggered by a time-based action in AWS lambda, modifying the inputs and outputs to be stored in an AWS S3 bucket, but for simplicity this approach is not taken here.

## 2.2 Prefect Instance
In In an Ubuntu Server 20.04 LTS or 22.04 LTS virtual machine with the [3_Deployment_Code/src/requirements.txt](3_Deployment_Code/src/requirements.txt) installed with the command

 `pip install -r requirements.txt`
 
The tested VM were:
- AWS EC2 (in a VM `t4g.large`).
- Azure Virtual machine (in a `D4pls v5`).

The execution steps for the Prefect Instance should follow:
1. Activation of local or remote MLFlow server (according to the section [3.1 of the present README](https://github.com/camilo-cf/mlops_car_sales#31-experiment-tracking-and-model-registry)).
2. Execution of Prefect Orion (UI):

      `prefect orion start &` 
     
     To run it in background.
3. Compilation of the prefect orchestrated pipeline: 
     
     `prefect deployment build 3_Deployment_Code/src/orchestration.py:car_purchase_prediction --name car_purchase_prediction --infra process`

     To run the orchestrated file locally.

4. Prefect apply and prefect agent schedule:

     `prefect deployment apply car_purchase_prediction-deployment.yaml`

     With the ID of the task add a manual quer with the desired schedule and execute the agent with the task_id (in this example 953962d2-2b7d-415d-be41-6643b60e8e75)

     `prefect agent start 953962d2-2b7d-415d-be41-6643b60e8e75`

**Note**: The execution process for each one of the deployment options will be detailed in the [section 4](https://github.com/camilo-cf/mlops_car_sales#4-reproducibility).

# 3. MLOps approach to tackle the problem
This project follows:
1. An Exploratory Data Analysis (EDA) presented in [1_EDA](./1_EDA). Where the environment can be installed by installing the `requirements.txt` in the given location [requirements.txt](./1_EDA/requirements.txt) in a virtual env with `pip install -r requirements.txt`. The EDA uses a pandas profiling report to ease the process.
2. The ML modeling section presented in [2_ML_modeling](2_ML_modeling). It shows:
     - The syntetic data creation for the hypothetical users in the [notebook](2_ML_modeling/0_create_fake_data.ipynb), saved in a CSV file. For simulate the bank users to evaluate.
     - The [modeling section](2_ML_modeling/1_modeling.ipynb), where is performed:
          - Source database load
          - Data split and preprocessing
          - Execution of multiple ML Classificarion models.
               - Decision Tree Model
               - Logistic Regression
               - XGBoost Classifier
          - Comparison of model performance, where the XGBoost performed better than others (89% of accuracy for the validation data).
          - Preparation of the deployment model (Scikitlearn Pipeline made by the StandardScaler and the Best Model)
          - Serialization and saving of the pipeline with the joblib library.
          - Tests on batch data with the saved pipeline on the syntetic data.
3. Implementation of:
     - Experiment tracking and model registry (MLFlow)          
     - Workflow orchestration (Prefect)
     - Model Monitoring (Evidently)
     - Best practices
          - Linting
          - CI/CD (TODO)

     The configuration of the environment follows the configuration provided by the [requirements.txt](3_Deployment_Code/src/requirements.txt) provided in the `3_Deployment_Code/src/requirements.txt`.

     Other details will be detailed in the upcoming subsections.

## 3.1 Experiment Tracking and model registry

### 3.1.1 Local Usage of MLFlow

In contrast with the [docker deployment version](3_Deployment_Code/src/orchestration_docker.py), the [experiment tracking version](3_Deployment_Code/src/orchestration.py) was developed to track the experiments and register the most promising models.

The implementation was made with **MLFlow 1.28.0**, tested on a local MLFlow server as it is possible to observe on the line 18 of [3_Deployment_Code/src/orchestration.py](3_Deployment_Code/src/orchestration.py). 

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")
```

To execute the server locally it will be required to execute:

`mlflow server --host 0.0.0.0 --port 5000   --backend-store-uri sqlite:///mlflow.db   --default-artifact-root $PWD/mlruns --serve-artifacts`

Where `sqlite:///mlflow.db` is the sqlite3 database created to persist the backend in the local machine. This execution will allow a local backend storage (Experiment tracking), a local artifact storage (Experiment tracking) and a local artifact serving (usage of the Model Registry).

### 3.1.2 AWS EC2 and AWS S3 Usage of MLFlow

On the other hand, this was tested as well with a remote MLFlow server in AWS EC2 (in a VM `t2.micro`) with MLFlow and sqlite3 installled, persisting the backend storage locally in the EC2 VM and the artifact storage in an AWS S3 bucket. Executing:

`mlflow server -h 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://example-desired-mlflow-bucket $PWD/mlruns --serve-artifacts &`

The MLFlow server will run in background pointing to the given AWS S3 bucket (with the required permisions), being able to track the experiments and serve the obtained models.

To execute the [3_Deployment_Code/src/orchestration.py](3_Deployment_Code/src/orchestration.py).  is required to have a connection with the AWS EC2 VM, needing to update the line 18 of [3_Deployment_Code/src/orchestration.py](3_Deployment_Code/src/orchestration.py). 

```python
mlflow.set_tracking_uri("remote_server_uri")
```
This can be achieved also with a local environment variable, and its update.

### 3.1.3 Execution Test of the MLFlow Model Registry 

In the [1_modeling.ipynb notebook](2_ML_modeling/1_modeling.ipynb) under the `Test execution model from Model Registry` section is tested the prediction of the best obtained model (by accuracy) saved for a given experiment name, in a given `server_uri` (in the case localhost). This example executes the prediction on a sample of synthetic data (simulation of bank users).

## 3.2 Workflow orchestration

The workflow orchestration is achieved using **Prefect 2.2.0**. The *@tasks* of the orchestration are possible to observe on the file  [3_Deployment_Code/src/orchestration.py](3_Deployment_Code/src/orchestration.py). 

The main tasks are functions under the [3_Deployment_Code/src/orchestration.py](3_Deployment_Code/src/orchestration.py) file:
- `read_csv2df`:  Read a CSV and returns a pandas DataFrame.
- `cross_validation`: Cross-validation execution from an external first-party (self-developed in) [library](3_Deployment_Code/src/libs/preprocess.py).
- `save_train_test_data`: Save in pkl the training and test data.
- `train_and_log_model`: From an external first-party (self-developed in) [library](3_Deployment_Code/src/libs/train.py) the train and test data is loaded from a pkl file and 3 ML models (DecisionTreeClassifier, ModelLogisticRegression and ModelXGBClassifier) are trainned and **tracked (metrics and artifacts) by MLFlow**.
- `register_model`: Verifies the all **tracked experiments** and selects the model with the best test accuracy to obtain its metadata and **register the model** in the MLFlow service.
- `run_model_save_pred`: Takes a registered model (the best is expected) and a given pandas dataframe, to apply the model in the dataframe and save its output in a CSV file that can be copied to an AWS S3 bucket with the `boto3` library to write on AWS S3, allowing a  remote access of the outu, being the output of the process (Expected to be read by the area that needs the predictions).

All the **@tasks** uses the **prefect logger** (*get_run_logger()*) to track the status and states of the execution and the obtained results.

The **@flow** is the assembling of the tasks, in this case corresponds to `car_purchase_prediction` function, where the **@tasks** are called in the right order as a **SequentialTaskRunner()**.

## 3.3 Model deployment
The deployment of the current model takes an offline batch approach that can be triggered by a time based rule. Different approaches were taken for this step:
1. **Cloud-based solution**:

     a. **Load a model saved in the MLFlow model registry orchestrated by Prefect**: A local deployment of Prefect (as explained in [section 2.2](https://github.com/camilo-cf/mlops_car_sales#22-prefect-instance)) that orchestrates a deployment of the process, deploying the registered model (in a MLFlow server that can be in AWS or local as explained in [section 3.1](https://github.com/camilo-cf/mlops_car_sales#31-experiment-tracking-and-model-registry)).
     
     b. **Load a model stored in the VM local storage as .joblib model**: A Docker container in AWS ECR triggered by a time-based rule in AWS Lambda for the deployment of a model saved in the default model folder (./model/model.joblib).

3.  **Local deployment**:

     a. **Load a model saved in the MLFlow model registry orchestrated by Prefect**: A local deployment of Prefect (as explained in [section 2.2](https://github.com/camilo-cf/mlops_car_sales#22-prefect-instance)) that orchestrates a deployment of the process, deploying the registered model (in a MLFlow server that can be in AWS or local as explained in [section 3.1](https://github.com/camilo-cf/mlops_car_sales#31-experiment-tracking-and-model-registry)).
     
     b. **Load a model stored in the VM local storage as .joblib model**: A Docker container scheduled with a `cronjob` in a local machine for the deployment of a model saved in the default model folder (./model/model.joblib).

## 3.4 Model monitoring
For the model monitoring used Evidently (evidently==0.1.57.dev0) was used to track the data drift of the input data. Its output is a HTML report with the drift analysis.

## 3.5 Best practices

### 3.5.1 Unit Tests
- Not implemented

### 3.5.2 Integration Tests
- Not implemented

### 3.5.3 Linting
- `pylint` was used as a linter 
- `black` was used as an autoformatter 

### 3.5.4 Makefile
- Not implemented

### 3.5.5 Pre-commit hooks
- Not implemented

### 3.5.6 CI/CD
A CI pipeline running in Github actions was implemented with pylint with a treshold of 6/10 to approve. The code is [here](.github/workflows/pylint.yml).

# 4. Reproducibility

# 4.1 Exploratory data analysis
1. Create a virtual environment. I like to use miniconda, so I create the environment as follows
     ```bash
     conda create -n eda python=3.9 -y
     ```
2. Activate the virtual environment
     ```bash
     conda activate eda
     ```
3. Then place yourself in the root of the directory of the project.

4. Install the `requirements.txt` located in the 1_EDA  folder:
     ```bash
     pip install -r 1_EDA/requirements.txt
     ```
5. Execute the `1_EDA/0_EDA.ipynb` notebook and check the results.

**Note:** Be sure you downloaded the dataset `car_data.csv` from [kaggle](https://www.kaggle.com/datasets/gabrielsantello/cars-purchase-decision-dataset?resource=download) and located it in `/data/car_data.csv` in the root project folder.

# 4.2 ML Modeling
1. Within the same virtual environment you created in the [4.1 section](https://github.com/camilo-cf/mlops_car_sales#41-exploratory-data-analysis). Execute the `2_ML_modeling/0_create_fake_data.ipynb` notebook to create the synthetic data.

2. Now, run the `2_ML_modeling/1_modeling.ipynb` in the same virtual environment without the **Test execution model from Model Registry**. You can check the modeling process and the predition draft.

# 4.3 Deployment Code
## 4.3.1 Prefect & MLFlow
With an AWS EC2 (`t4g.large`) OR an Azure Virtual machine (`D4pls v5`) with Ubuntu Server 20.04 LTS or 22.04 LTS; OR in a local machine with similar features. 

Be sure python 3.9 is installed on it. You can install miniconda and follow the steps 1 and 2, or skip them if you don't.

1. Create a virtual environment. I like to use miniconda, so I create the environment as follows
     ```bash
     conda create -n mlops_deploy python=3.9 -y
     ```
2. Activate the virtual environment
     ```bash
     conda activate mlops_deploy
     ```
3. Then place yourself in the root of the directory of the project.

4. Install the `requirements.txt` located in the 3_Deployment_Code  folder:
     ```bash
     pip install -r 3_Deployment_Code/src/requirements.txt
     ```  
5. Run Prefect UI (in background).
     ```bash
     prefect orion start &
     ```

6. In the same server or another one (after installing `mlflow==1.28.0` in a python environment or just the base env), run the MLFlow server after setting it as was presented in the MLOps zoomcamp course. 

     This will run the server in a local artifact-root with the possibility to serve artifacts. The backend data will be saved in `mlflow.db`. And the command will run in background (your terminal will be free to be used).

     ```bash
     mlflow server --host 0.0.0.0 --port 5000   --backend-store-uri sqlite:///mlflow.db   --default-artifact-root $PWD/mlruns --serve-artifacts &
     ```
7. Compile the Prefect orchestration script.
     
     ```bash
     prefect deployment build 3_Deployment_Code/src/orchestration.py:car_purchase_prediction --name car_purchase_prediction --infra process
     ```

     If all is well it should return something like:
     ```bash
     Found flow 'car-purchase-prediction'

     Deployment YAML created at '/home/user/.../car_purchase_prediction-deployment.yaml'.
     ```
     
8. Apply the detected flow in the current Prefect instance.
     ```bash
     prefect deployment apply car_purchase_prediction-deployment.yaml
     ```
     
     If all is well it should return something like (*the id can change*):

     ```bash
     Successfully loaded 'car_purchase_prediction'
     Deployment 'car-purchase-prediction/car_purchase_prediction' successfully created with id 'fd0c0109-f004-4b79-8dad-c6bd60866e38'.

     To execute flow runs from this deployment, start an agent that pulls work from the the 'default' work queue:
     $ prefect agent start -q 'default'
     ```

9. On the browser access to the Prefect server (in my case) http://127.0.0.1:4200/deployments (and as I'm accessing by VS Code remote SSH it redirects me from the console - or open the VM ports to access it remotely) and click on Edit.

     ![Prefect deployment](/docs/img/4_3_1_9.PNG.jpg "Prefect deployment image")

10. Add the desired scheduler (in my case each 24h with cron it should look like `0 0 * * *` to run everyday at 00:00 - verify your cron expresions [here](https://crontab.guru/)).

     ![Prefect schedule](/docs/img/4_3_1_10.PNG.jpg "Prefect schedule image")
     ![Prefect schedule](/docs/img/4_3_1_10a.PNG.jpg "Prefect schedule image")
     
     Set `0 0 * * *`.

     Save.

11. On the browser access to the Prefect server (in my case) http://127.0.0.1:4200/runs (and as I'm accessing by VS Code remote SSH it redirects me from console) or open the VM ports to access it remotely, you will see the scheduled tasks.

     ![Prefect schedule](/docs/img/4_3_1_11.jpg "Prefect schedule image")

12. Create the Prefect Agent to run the scheduled task with the previously obtained id (add a ` &` at the end if you want it runs in background). Follow the suggestion obtained from step 8.
     ```bash
     prefect agent start -q 'default'
     ```

     The agent will execute the given tasks when the scheduled time will be achieved. The executed tasks can be observed here http://127.0.0.1:4200/runs (in my case), the executed tasks shows in green.

     ![Prefect schedule](/docs/img/4_3_1_12.jpg "Prefect schedule image")
     
     You can also verify the MLFlow tracking and registered models (in my case the Experiment tracking in http://127.0.0.1:5000/#/experiments/1 and the Model Registry http://127.0.0.1:5000/#/models)

     ![MLFlow experiment tracking](/docs/img/4_3_1_12a.jpg "MLFlow experiment tracking")

     ![MLFlow model registry](/docs/img/4_3_1_12b.jpg "MLFlow model registry")


13. Explore the Prefect and MLFlow UI

     ![Prefect details](/docs/img/4_3_1_13a.jpg "Prefect details")
     ![Prefect details](/docs/img/4_3_1_13b.jpg "Prefect details")


14. The Model monitoring is based on the datadrift of Evidently, and it is temporarly recorded as a MLFlow experiment (without metrics) and it can be easily accessed. As well the model output is recorded (temporarly) also as an artifact just to keep track of it.

     ![ML artifacts](/docs/img/4_3_1_14a.jpg "ML artifacts")
     ![Model Drift](/docs/img/4_3_1_14b.jpg "Model Drift")


## 4.3.2 Docker execution (in a VM)
1. Be sure docker is running properly.
2. Execute the bash file to run the Docker container
     ```bash
     bash run_docker.sh
     ```
3. If it was successful check the output of the file in the local folder.

## Disclaimer
This is an academic exercise, merely for educational purposes.