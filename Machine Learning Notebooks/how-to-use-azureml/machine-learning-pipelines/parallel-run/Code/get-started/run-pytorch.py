# run-pytorch.py
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig
import os

if __name__ == "__main__":
    # create a workspace
    subscriptionId = 'bfb59099-69db-4d2b-887e-abcf6ccdb5c4'
    resourceGroup = 'AZR-C11-DV-122-01-RnD'
    workspaceName = 'lsmltraining'

    curr_dir = f'{os.getcwd()}\\Machine Learning Notebooks\\how-to-use-azureml\\machine-learning-pipelines\\parallel-run\\Code\\get-started'

    ws = Workspace(subscription_id=subscriptionId,resource_group=resourceGroup,workspace_name=workspaceName)
    experiment = Experiment(workspace=ws, name='day1-experiment-train')
    config = ScriptRunConfig(source_directory=f'{curr_dir}\\src',
                             script='train.py',
                             compute_target='liverpoolcluster')

    # set up pytorch environment
    env = Environment.from_conda_specification(
        name='pytorch-env',
        file_path=f'{curr_dir}\\pytorch-env.yml'
    )
    config.run_config.environment = env

    run = experiment.submit(config)

    aml_url = run.get_portal_url()
    print(aml_url)
