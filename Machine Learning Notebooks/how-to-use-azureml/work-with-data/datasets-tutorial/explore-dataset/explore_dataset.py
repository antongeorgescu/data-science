# azureml-core of version 1.0.72 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = 'bfb59099-69db-4d2b-887e-abcf6ccdb5c4'
resource_group = 'AZR-C11-DV-122-01-RnD'
workspace_name = 'lsmltraining'

workspace = Workspace(subscription_id, resource_group, workspace_name)
print(workspace.name)
dataset = Dataset.get_by_name(workspace, name='TD-Auto_Prices_Prediction_Pipeline-Clean_Missing_Data-Cleaning_transformation-f6dc0eb1')
dataset.download(target_path='C:\\temp', overwrite=False)
