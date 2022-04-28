import pandas as pd
import os
from azureml.core import Workspace, Dataset

# create sample dataframe
df = pd.DataFrame({'num_legs': [2, 4, 8, 0, 2],
                'num_wings': [2, 0, 0, 0, 0],
                'num_specimen_seen': [10, 2, 1, 8, 5]},
                index=['falcon', 'dog', 'spider', 'fish','kangaroo'])

# save dataframe to local path
local_dir = f'{os.getcwd()}\\data_samples'
local_path = f'{local_dir}\\few_creatures.csv'
df.to_csv(local_path)

# create a workspace
subscriptionId = 'bfb59099-69db-4d2b-887e-abcf6ccdb5c4'
resourceGroup = 'AZR-C11-DV-122-01-RnD'
workspaceName = 'lsmltraining'

workspace = Workspace(subscription_id=subscriptionId,resource_group=resourceGroup,workspace_name=workspaceName)
# get the datastore to upload prepared data
datastore = workspace.get_default_datastore()
# upload the local file from src_dir to the target_path in datastore
datastore.upload(src_dir=local_dir, target_path='data_samples')

# create a dataset referencing the cloud location
ds_creatures = Dataset.Tabular.from_delimited_files(datastore.path('data_samples/few_creatures.csv'))

# register dataset
ds_creatures.register(workspace, 'five_creatures', description='Sample with a few creatures made by God',create_new_version=True)

print('List of all registered datasets')
for dsname in Dataset.get_all(workspace):
    print(dsname)
