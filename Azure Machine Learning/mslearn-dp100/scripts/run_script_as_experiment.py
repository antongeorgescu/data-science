import azureml.core as azc
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.widgets import RunDetails
import os,shutil

PROJECT_DATA_FOLDER = "mslearn-dp100/data"
EXPERIMENT_SCRIPTS_FOLDER = "mslearn-dp100/scripts/experiment_scripts"

# Copy the data file into the experiment folder
shutil.copy(f'{PROJECT_DATA_FOLDER}/TACN-original.csv', os.path.join(f'{EXPERIMENT_SCRIPTS_FOLDER}/data', "TACN-original.csv"))

ws = Workspace.from_config(path='mslearn-dp100',_file_name='config_mlresource-aml.json')
print('Ready to use Azure ML {} with workspace {}'.format(azc.VERSION, ws.name))

# Create a Python environment for the experiment
# sklearn_env = Environment.from_conda_specification("exp-base", "mslearn-dp100/environment.yml")
sklearn_env = Environment.from_pip_requirements("exp-base", "mslearn-dp100/environment.yml")

# Ensure the required packages are installed (we need scikit-learn, Azure ML defaults, and Azure ML dataprep)
packages = CondaDependencies.create(conda_packages=['scikit-learn','pip'],
                                    pip_packages=['azureml-defaults','azureml-dataprep[pandas]','matplotlib'])
sklearn_env.python.conda_dependencies = packages

# Create a Python environment for the experiment (from a .yml file)
# sklearn_env = Environment.from_conda_specification("exp-base", "mslearn-dp100/environment.yml")

# Create a script config
script_config = ScriptRunConfig(source_directory=EXPERIMENT_SCRIPTS_FOLDER,
                                script='exp_sample_dataset.py',
                                environment=sklearn_env) 

# submit the experiment
experiment = Experiment(workspace = ws, name = 'experiment-pysdk-by-script')
run = experiment.submit(config=script_config)

# RunDetails(run).show()

run.wait_for_completion(show_output=True)

# Get logged metrics
print(f'Metrics for experiment ID:{run.id}, Name:{run.display_name}')
metrics = run.get_metrics()
for key in metrics.keys():
    print(key, metrics.get(key))
print('\n')
for file in run.get_file_names():
    print(file)

run.get_details_with_logs()

# Download all files
log_folder = f'{EXPERIMENT_SCRIPTS_FOLDER}/logs'
run.get_all_logs(destination=log_folder)

# Verify the files have been downloaded
for root, directories, filenames in os.walk(log_folder): 
    for filename in filenames:  
        print (os.path.join(root,filename))

# get experiment running history
my_experiment = ws.experiments['experiment-pysdk-by-script']
for logged_run in my_experiment.get_runs():
    print('Run ID:', logged_run.id,' Run name:',logged_run.display_name,' Status:',logged_run.status)
    metrics = logged_run.get_metrics()
    for key in metrics.keys():
        print('-', key, metrics.get(key))
