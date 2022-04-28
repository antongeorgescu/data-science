from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
import os

# create a workspace
subscriptionId = 'bfb59099-69db-4d2b-887e-abcf6ccdb5c4'
resourceGroup = 'AZR-C11-DV-122-01-RnD'
workspaceName = 'lsmltraining'

ws = Workspace(subscription_id=subscriptionId,resource_group=resourceGroup,workspace_name=workspaceName)
experiment = Experiment(workspace=ws, name='day1-experiment-hello')
myenv = Environment.get(workspace=ws, name="AzureML-Minimal")

script_dir = f'{os.getcwd()}\\Machine Learning Notebooks\\how-to-use-azureml\\machine-learning-pipelines\\parallel-run\\Code\\first-experiment\\src'
config = ScriptRunConfig(source_directory=script_dir, script='hello.py', compute_target='liverpoolcluster',environment=myenv)

run = experiment.submit(config)
aml_url = run.get_portal_url()
print(aml_url)
