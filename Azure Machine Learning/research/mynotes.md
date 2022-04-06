# Azure ML

* TabularExplainer calls one of the three SHAP explainers (TreeExplainer, DeepExplainer, or Kernel Explainer.) TabularExplainer automatically selects the most appropriate one for your use case
* For any batch inference service deployed using Azure ML Designer, default configuration requires an authentication header to be passed as the headers parameter for requeAzure Container Intsasssssssssts 
* Calling AlsWebservice.deploy_configuration without any parameters will enable key-authentication by default
* Early termination:<br/>
    + <b>Truncation Selection</b> policy: cancels % runs with low performance on the primary metric for a given evaluation interval<br/>
    + <b>Bandit</b> policy: is based on slack factor/slack amount and evaluation interval. Bandit ends runs when the primary metric isn't within the specified slack factor/slack amount of the most successful run.Slack_factor or slack_amount is the slack allowed with respect to the best performing training run.<br/>
    + <b>Median</b> policy: is an early termination policy based on running averages of primary metrics reported by the runs. This policy computes running averages across all training runs and stops runs whose primary metric value is worse than the median of the averages.
* A compute target is a designated compute resource or environment where you run your training script or host your service deployment. This location might be your local machine or a cloud-based compute resource. Using compute targets makes it easy for you to later change your compute environment without having to change your code.
    + Azure Container Instances (ACI) and Local web service are deployment targets that provide low cost instances that can be used for testing and debugging CPU based workloads
    + Azure Kubernetes Services (AKS) is used for production workloads. AKS provides fast response times and autoscaling of deployed service, but it's costly compared with ACIand local web services. AKS is not suitable for testing and debugging.
    + Azure ML compute clusters are used for batch inference pipelines.
* Primary metric spearman_correlation" is used for regression tasks and works on numeric and logical data only
* Azure ML Designer: Publish button does not deploy the model as a web service endpoint. To deploy the model as a web service endpoint, you need to navigate to Models menu on ML studio, select the model you want to deploy, and click Deploy button. 
    + Publish button creates a REST endpoint to the pipeline that other users/developers/data scientists can make calls to, Ir provides an endpoint with a key-based authentication
    + Publish button does not run the pipeline. To run the pipeline against test dataset, whihc is part of the model, you need to click Submit on designer canvas
* TabularDatasets represeent collections of structured data, like the ones found in a CSV file. You can create a TabularDataset and reference it directly in a training script without having to register the dataset with an AML workspace
* To automate the creation of Azure ML workspace using Azure CLI:
    + Install Azure CLI runtime
    + Log into Azure
    + Select the correct subscription
    + Run az extension add -n azure-cli-ml to register ML extension to run az ml commands
    + Run az group create --name <resource-group-name> --location <location> to create resource group for WS
    + Run az ml workspace create -w <workspace-name> -g <resource-group-name> to create WS and attached to resource group
* Use key-authentication only for ACI compute targets. ACI only supports key-authentication
* In addition to supporting multiple-node clusters, AKS can be used for experiments that require H/W acceleration via GPU or Field-Programmable Gate Arrays (FPGA)
* AKS can dynamically scale compute availability based on workload
* Global surrogate model is meant to be an interpretable approximation of a black box model.Black box modela are those for which no explanation exists, whihc means the public does not know how the model makes its predictions. Once a surrogate model is trained, the mimic explainer interpretability technique can be used to interpret the model
* Permutation Feature Importance (PFI) explainer randomly shuffles features during model training and then calculates the impact on model performance
* SHAP is a model-specific interpretability technique used for linear models. 
* SHAP explainers use calculations based on coallition game theory
* Local logging during training process (use parameter show_output)
```
        from azureml.core import Experiment
        exp = Experiment(ws,experiment_name)
        run = exp.submit(config=run_config_object,show_output=True)
```
* Logging run-related data within experiment (show method start_logging())
```
        from azureml.core import Experiment
        exp = Experiment(ws,experiment_name)
        run = exp.start_logging()
        run.log("temperature",35)
```
* Data Labeling projects
    + AML datasets with labels are called <b>labeled datasets</b>. These specific datasets are TabularDatasets with a dedicated label column and are only created as an output of Azure ML data labeling projects.
    + Create a <b>data labeling project</b> for image labeling or text labeling.
    + Machine Learning supports data labeling projects for image classification 
    + When complete a data labeling project, you can export the label data from a labeling project.Doing so allows you to capture both the reference to the data and its labels, and export them in either 1) COCO format or 2) Azure ML dataset
    + COCO file is created in the default blob store of Azure ML workspace in a folder within <i>export/coco</i>
* Azure ML Designer is only available with the enterprise edition of the workspace
* When creating a Workspace object, parameter <i>private_endpoint_auto_approval</i> is True by default and has no need to be specified. 
```
        Workspace.create(name = workspaceName,
                        subscription_id = subscriptionId,
                        resource_group = resourceGroup,
                        private_endpoint_config = privateEndpointConfig,
                        sku = 'enterprise')
```
* For the code below, you cannot specify continuous increase of batch_size, by using for instance a uniform(0.05,01) function. GridParameterSampling does a grid search over all possible values defined in the search space.
However, you can create discrete hyperparameters using a distribution (like a range object.)
```
        from azureml.train.hyperdrive import GridParameterSampling
        from azureml.train.hyperdrive import choice

        param_sampling = GridParameterSampling({
            "num_hidden_layers": choice(1,2,3),
            "batch_size": choice(16,32,64)
        })
```

### Practice Set Last Question: 44 of 54<hr/>


# Links: 
1. https://christophm.github.io/interpretable-ml-book/ 
2. https://shap.readthedocs.io/en/latest/ 
3. https://christophm.github.io/interpretable-ml-book/shapley.html
4. https://github.com/MicrosoftLearning/DP100/blob/master/07B%20-%20Creating%20a%20Batch%20Inferencing%20Service.ipynb
5. https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets
6. https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-environments
7. https://azure.github.io/azureml-sdk-for-r/reference/register_azure_blob_container_datastore.html
8. https://docs.microsoft.com/en-us/azure/databricks/clusters/
9. https://docs.microsoft.com/en-us/python/api/azureml-pipeline-core/azureml.pipeline.core.schedule(class)?view=azure-ml-py
10. https://docs.microsoft.com/en-us/azure/machine-learning/concept-designer
11. https://docs.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication
12. https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-interpretability
13. https://docs.microsoft.com/en-us/azure/machine-learning/how-to-track-monitor-analyze-runs?tabs=python
14. https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-labeled-dataset
15. https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters


