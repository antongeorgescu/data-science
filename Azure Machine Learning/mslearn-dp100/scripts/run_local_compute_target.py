from azureml.core import Workspace
from azureml.core import Experiment
import pandas as pd
from azureml.widgets import RunDetails
import json

ws = Workspace.from_config()

# for compute_name in ws.compute_targets:
#     compute = ws.compute_targets[compute_name]
#     print(compute.name, ":", compute.type)

# create an experiment variable
experiment = Experiment(workspace = ws, name = "experiment-with-python-sdk")

# start the experiment
run = experiment.start_logging()

# load the dataset and count the rows
data = pd.read_csv('mslearn-dp100/data/tacn-train.csv')
row_count = (len(data))

# Log the row count
run.log('observations', row_count)

# end the experiment
run.complete()

# Get logged metrics
metrics = run.get_metrics()

metrics_json = json.dumps(metrics, indent=2)
print(metrics_json)

# the json file where the output must be stored
metrics_file_path = "mslearn-dp100/outputs/metrics.json"
out_file = open(metrics_file_path, "w")
json.dump(metrics_json,out_file)

# RunDetails(run).show()

# upload metrics file in AML workspaceblob
run.upload_file(name='outputs/experiment-pysdk-metrics.json', path_or_stream=metrics_file_path)
