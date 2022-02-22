import azureml.core as azc
from azureml.core import Experiment, ScriptRunConfig, Environment, Workspace
from azureml.widgets import RunDetails

# Load the workspace from the saved config file
ws = Workspace.from_config(path='mslearn-dp100',_file_name='config_mlresource-aml.json')
print('Ready to use Azure ML {} to work with {}'.format(azc.VERSION, ws.name))

# Create a Python environment for the experiment (from a .yml file)
an_env = Environment.from_conda_specification("mlflow_experiment_env", "mslearn-dp100/environment.yml")

experiment_folder = 'mslearn-dp100\scripts\experiment_scripts'

# Create a script config
script_mlflow = ScriptRunConfig(source_directory=experiment_folder,
                                script='local_mlflow_diabetes.py',
                                environment=an_env) 

# submit the experiment
experiment = Experiment(workspace=ws, name='local-mslearn-diabetes-mlflow')
run = experiment.submit(config=script_mlflow)
RunDetails(run).show()
run.wait_for_completion()
