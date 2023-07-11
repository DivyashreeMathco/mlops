# Databricks notebook source
import mlflow

# COMMAND ----------

mlflow.set_experiment("untuned_random_forest")

# COMMAND ----------

def read(database_name="MLOPS", schema_name="RAW_DATA", table_name=str):
    
    options = {
        "sfUrl": "gx89677.central-india.azure.snowflakecomputing.com",
        "sfUser": "Divyasnowflake",
        "sfPassword": "Divya@1525",
        "sfDatabase": database_name,
        "sfSchema": schema_name,
        "sfWarehouse": "COMPUTE_WH"}
    
    df1 = spark.read.format("snowflake").options(**options).option("query", f"select * from {table_name};").load()
    
    #df2 = spark.read.format("snowflake").options(**options).option("query", f"select * from {table_name};").load()
    return df1


# def snowflake(read):
#     df1 = read(database_name="MLOPS", schema_name="RAW_DATA", table_name="white_wine")

#     #df2 = read(database_name="MLOPS", schema_name="RAW_DATA", table_name="red_wine")
#     return df1

# COMMAND ----------

if __name__ == "__main__":
    df1 = read(table_name="white_wine")
    df2 = read(table_name="red_wine")
    display(df1)
    display(df2)

# COMMAND ----------

# from snowflake import read
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def EDA():
    df1 = read(database_name="MLOPS", schema_name="RAW_DATA", table_name="white_wine")

    df2 = read(database_name="MLOPS", schema_name="RAW_DATA", table_name="red_wine")

    df1 = df1.toPandas()
    df1 = df1.rename(columns=lambda x: x.replace('"', ''))
    df2 = df2.toPandas()
    df2 = df2.rename(columns=lambda x: x.replace('"', ''))
    df2['is_red'] = 1
    df1['is_red'] = 0
    data = pd.concat([df2, df1], axis=0)
    data['quality'].unique()
    data['quality']
    
    sns.distplot(data.quality, kde=False)
    high_quality = (data.quality >= 7).astype(int)
    data.quality = high_quality
    data['quality']
 
    dims = (3, 4)
    
    f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
    axis_i, axis_j = 0, 0
    for col in data.columns:
        if col == 'is_red' or col == 'quality':
            continue # Box plots cannot be used on indicator variables
        sns.boxplot(x=high_quality, y=data[col], ax=axes[axis_i, axis_j])
        axis_j += 1
        if axis_j == dims[1]:
            axis_i += 1
            axis_j = 0

        data.isna().any()
        new_data = spark.createDataFrame(data)
        new_data.write.format("snowflake").option("dbtable", "EDA_DATA").option("sfUrl", "gx89677.central-india.azure.snowflakecomputing.com").option("sfUser", "Divyasnowflake").option("sfPassword", "Divya@1525").option("sfDatabase", "MLOPS").option("sfSchema", "REFINED_DATA").option("sfWarehouse", "COMPUTE_WH").mode("append").save()
    return data


# COMMAND ----------

dt = EDA()

# COMMAND ----------

dt

# COMMAND ----------

from sklearn.model_selection import train_test_split

def training(data):
 
    X = data.drop(["quality"], axis=1)
    y = data.quality
    
    # Split out the training data
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=123)
    
    # Split the remaining data equally into validation and test
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.4, random_state=123)
    return X_train, X_test, y_train, y_test

# COMMAND ----------

X_train, X_test, y_train, y_test = training(dt)

# COMMAND ----------

X_train

# COMMAND ----------

y_train

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time
 
# The predict method of sklearn's RandomForestClassifier returns a binary classification (0 or 1). 
# The following code creates a wrapper function, SklearnModelWrapper, that uses 
# the predict_proba method to return the probability that the observation belongs to each class. 
 
class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]
 
# mlflow.start_run creates a new MLflow run to track the performance of this model. 
# Within the context, you call mlflow.log_param to keep track of the parameters used, and
# mlflow.log_metric to record metrics like accuracy.
  def mlflow_run(X_train,y_train,X_test,y_test):
    with mlflow.start_run(run_name='untuned_random_forest'):
        n_estimators = 10
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
        model.fit(X_train, y_train)
        
        # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
        predictions_test = model.predict_proba(X_test)[:,1]
        auc_score = roc_auc_score(y_test, predictions_test)
        mlflow.log_param('n_estimators', n_estimators)
        # Use the area under the ROC curve as a metric.
        mlflow.log_metric('auc', auc_score)
        wrappedModel = SklearnModelWrapper(model)
        # Log the model with a signature that defines the schema of the model's inputs and outputs. 
        # When the model is deployed, this signature will be used to validate inputs.
        signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
        
        # MLflow contains utilities to create a conda environment used to serve models.
        # The necessary dependencies are added to a conda.yaml file which is logged along with the model.
        conda_env =  _mlflow_conda_env(
                additional_conda_deps=None,
                additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
                additional_conda_channels=None,
            )
        mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)
        return model

# COMMAND ----------

wrappedModel = SklearnModelWrapper.mlflow_run(X_train,y_train,X_test,y_test)

# COMMAND ----------

wrappedModel

# COMMAND ----------

def feature_importances(model,X_train):
    feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
    feature_importances.sort_values('importance', ascending=False)
    return feature_importances

# COMMAND ----------

fet_imp=feature_importances(wrappedModel,X_train)

# COMMAND ----------

fet_imp

# COMMAND ----------

from mlflow.tracking import MlflowClient
def Registering():
    run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_random_forest"').iloc[0].run_id
    model_name = "quality_prediction"
    model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)
 
# Registering the model takes a few seconds, so add a small delay
    time.sleep(15)

    client = MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production",
    )

    model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
 
# Sanity-check: This should match the AUC logged by MLflow
    print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')
    return model, model_name

# COMMAND ----------

model, model_name = Registering()

# COMMAND ----------

model_name

# COMMAND ----------

import mlflow.pyfunc
from pyspark.sql.functions import struct

def inferencing(X_train,model_name):
    spark_df = spark.createDataFrame(X_train)
    table_path = "dbfs:/msdr000@gmail.com/delta/wine_data"
    dbutils.fs.rm(table_path, True)
    spark_df.write.format("delta").save(table_path)

    apply_model_udf = mlflow.pyfunc.spark_udf(spark, f"models:/{model_name}/production")

    new_data = spark.read.format("delta").load(table_path)
 
    # Apply the model to the new data
    udf_inputs = struct(*(X_train.columns.tolist()))
    
    new_data = new_data.withColumn(
        "prediction",
        apply_model_udf(udf_inputs)
    )
    return new_data

# COMMAND ----------

inference = inferencing(X_train,model_name)

# COMMAND ----------

display(inference)

# COMMAND ----------

import os
os.environ["DATABRICKS_TOKEN"] = "dapic3c040da4c38bfd698ba3d34996babde-3"

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://adb-7827998753449886.6.azuredatabricks.net/model/quality_prediction/1/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
  #ds_dict = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  ds_dict = {"dataframe_split": dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

num_predictions = 5
served_predictions = score_model(X_test[:num_predictions])
model_evaluations = model.predict(X_test[:num_predictions])
served_predictions = np.array(list(served_predictions.values()))
# Compare the results from the deployed model and the trained model
pd.DataFrame({
  "Model Prediction": model_evaluations,
  "Served Model Prediction": served_predictions.flatten()
})

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json
    
os.environ["DATABRICKS_TOKEN"] = "dapic3c040da4c38bfd698ba3d34996babde-3"

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://adb-7827998753449886.6.azuredatabricks.net/serving-endpoints/wine/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
  #ds_dict = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  ds_dict = {"dataframe_split": dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

def predict(X_test,model):
    num_predictions = 5
    served_predictions = score_model(X_test[:num_predictions])
    model_evaluations = model.predict(X_test[:num_predictions])
    served_predictions = np.array(list(served_predictions.values()))
    # Compare the results from the deployed model and the trained model
    prediction_data = pd.DataFrame({
      "Model Prediction": model_evaluations,
      "Served Model Prediction": served_predictions.flatten()
    })

    prediction_data = spark.createDataFrame(prediction_data)

    prediction_data.write.format("snowflake").option("dbtable", "prediction").option("sfUrl", "gx89677.central-india.azure.snowflakecomputing.com").option("sfUser", "Divyasnowflake").option("sfPassword", "Divya@1525").option("sfDatabase", "MLOPS").option("sfSchema", "serving_score").option("sfWarehouse", "COMPUTE_WH").mode("append").save()
    
    return prediction_data

# COMMAND ----------

prediction_data = predict(X_test,wrappedModel)

# COMMAND ----------

display(prediction_data)

# COMMAND ----------

mlflow.__version__
