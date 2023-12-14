#!/usr/bin/env python
# coding: utf-8

# In[9]:

# jdbgjksdbgkjdsbgdksjgbdskjg
import os
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# Load iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Get the current working directory
cwd = os.getcwd()

# Create a subdirectory for MLflow
mlflow_dir = os.path.join(cwd, "mlruns")

# Check if the directory exists and is accessible
if os.access(mlflow_dir, os.R_OK):
    print(f"The directory {mlflow_dir} exists and is accessible.")
else:
    print(f"The directory {mlflow_dir} does not exist or is not accessible.")
    # Create the directory if it doesn't exist
    os.makedirs(mlflow_dir, exist_ok=True)


# In[ ]:


# Set the tracking URI to the MLflow directory
mlflow.set_tracking_uri('file://' + mlflow_dir)

# Set the tracking URI to the local tracking server
mlflow.set_tracking_uri('http://localhost:5000')

# Define the experiment name
experiment_name = "iris_experiment"

# Check if the experiment exists
experiment = mlflow.get_experiment_by_name(experiment_name)

# Check if the experiment exists
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    # If the experiment does not exist, create it
    mlflow.create_experiment(experiment_name)


# Set the experiment
mlflow.set_experiment(experiment_name)

# Start a new MLflow run
with mlflow.start_run(run_name="Iris_DT_Experiment"):
    # Define and train the model
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Make predictions
    predictions = clf.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')

    # Log model
    mlflow.sklearn.log_model(clf, "model")

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1", f1)

    print(f"Model accuracy: {accuracy}")
    print(f"Model F1 score: {f1}")


# In[ ]:




