# Azure ML Highlights

## Notes and Comments
* BEST SUMMARY on How Azure Machine Learning works: Architecture and concepts https://docs.microsoft.com/en-us/azure/machine-learning/concept-azure-machine-learning-architecture#endpoints
* TabularExplainer calls one of the three SHAP explainers (TreeExplainer, DeepExplainer, or Kernel Explainer.) TabularExplainer automatically selects the most appropriate one for your use case
* For any batch inference service deployed using Azure ML Designer, default configuration requires an authentication header to be passed as the headers parameter for request  
* Calling AksWebservice.deploy_configuration without any parameters will enable key-authentication by default
    + AksWebservice.deploy_configuration(token_auth_enabled=True,auth_enabled=False) is calling token-based authn
    + AksWebservice.deploy_configuration(auth_enabled=True) and AksWebservice.deploy_configuration() are both calling key-based authn
* Right way to call a batch inferencing model:
```
        interactive_auth = InteractiveLoginAuthentication()
        auth_header = interactive_auth.get_authentication_header()
        response = requests.post(rest_endpoint,
                                headers = auth_header,
                                json = {"ExperimentName" : "Batch_Pipeline_via_REST"})
```
* Early termination:<br/>
    + <b>Truncation Selection</b> policy: cancels % runs with low performance on the primary metric for a given evaluation interval<br/>
    + <b>Bandit</b> policy: is based on slack factor/slack amount and evaluation interval. Bandit ends runs when the primary metric isn't within the specified slack factor/slack amount of the most successful run.Slack_factor or slack_amount is the slack allowed with respect to the best performing training run.<br/>
    + <b>Median</b> policy: is an early termination policy based on running averages of primary metrics reported by the runs. This policy computes running averages across all training runs and stops runs whose primary metric value is worse than the median of the averages.
* A compute target is a designated compute resource or environment where you run your training script or host your service deployment. This location might be your local machine or a cloud-based compute resource. Using compute targets makes it easy for you to later change your compute environment without having to change your code.
    + Azure Container Instances (ACI) and Local web service are deployment targets that provide low cost instances that can be used for testing and debugging CPU based workloads
    + Azure Kubernetes Services (AKS) is used for production workloads. AKS provides fast response times and autoscaling of deployed service, but it's costly compared with ACIand local web services. AKS is not suitable for testing and debugging.
    + Azure ML compute clusters are used for batch inference pipelines.
* 
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
* Create an Azure ML workspace using Azure Cloud Shell
```
        az ml workspace create -n AML-Workspace -g AML-LearningResources
```
* An AutoMLConfig object with sparse data cannot have featurization urend on (ie "auto") and must be turned "off"
* If "primary_metric" parameter is **not specified** for AutoMLConfig class, *accuracy* is used for *classification* tasks, *normalized root mean squared* is used for *forecasting and regression* tasks, *accuracy* is used for *image classification and image multi label classification*, and *mean average* precision is used for *image object detection*.
* There are 4 ways to authenticate against Azure ML resources and workflows:
    + **Interactive**: You use your account in Azure Active Directory to either directly authenticate, or to get a token that is used for authentication. Interactive authentication is used during experimentation and iterative development. Interactive authentication enables you to control access to resources (such as a web service) on a per-user basis.
    + **Service principal**: You create a service principal account in Azure Active Directory, and use it to authenticate or get a token. A service principal is used when you need an automated process to authenticate to the service without requiring user interaction. For example, a continuous integration and deployment script that trains and tests a model every time the training code changes.
    + **Azure CLI session**: You use an active Azure CLI session to authenticate. Azure CLI authentication is used during experimentation and iterative development, or when you need an automated process to authenticate to the service using a pre-authenticated session. You can log in to Azure via the Azure CLI on your local workstation, without storing credentials in Python code or prompting the user to authenticate. Similarly, you can reuse the same scripts as part of continuous integration and deployment pipelines, while authenticating the Azure CLI with a service principal identity.
    + **Managed identity**: When using the Azure Machine Learning SDK on an *Azure Virtual Machine*, you can use a managed identity for Azure. This workflow allows the VM to connect to the workspace using the managed identity, without storing credentials in Python code or prompting the user to authenticate. Azure Machine Learning compute clusters can also be configured to use a managed identity to access the workspace when training models.

    <u>Note</u>: The Azure CLI commands in this article **require the azure-cli-ml**, or v1, extension for Azure Machine Learning. The enhanced v2 CLI (preview) using the ml extension is now available and recommended. To find which extensions you have installed, use **az extension list**. If the list of Extensions contains azure-cli-ml, you have the correct extension for the steps in this article.

* **Azure Machine Learning REST API** allow development of clients that use REST calls to work with the service. Details under https://docs.microsoft.com/en-us/rest/api/azureml/

* **Azure Machine Learning SDK for Python** very well summarized under https://docs.microsoft.com/en-ca/python/api/overview/azure/ml/?view=azure-ml-py

* **Comprehensive Guide in Using Azure Machine Learning** https://www.analyticsvidhya.com/blog/2021/09/a-comprehensive-guide-on-using-azure-machine-learning/

* Azure Machine learning studio supports *only supervised machine learning* models where we have training data and known labels

* **Collect data from models in production** https://docs.microsoft.com/en-us/azure/machine-learning/how-to-enable-data-collection

* **Azure Machine Learning compute clusters** are scalable ML platforms that consist of one or more CPU or GPU nodes. Compute clusters can scale from zero to hundreds of nodes, depending on workload. Compute clusters support the use of low-priority virtual machines (VMs) whihc do not have guaranteed availability.

* **Assign Azure roles using Azure CLI** https://docs.microsoft.com/en-us/azure/role-based-access-control/role-assignments-cli

* In the following code, the scheduler kicks in the pipeline everytime it detects changes in the datastore. Datastore is polled every 5 minutes to check changes.
```
        from azureml.pipeline.core import Schedule
        from azureml.core.datastore import Datastore

        datasetore = Datastore(workspace=ws,name="workspaceblobstore")
        schedule = Schedule.create(ws,name="TestSchedule",
                                    pipeline_id="pipeline_id",
                                    experiment_name="Hellow World",
                                    datastore=datastore,
                                    polling_interval=5,
                                    path_on_datastore="file/path")
```

* There are 2 different deploys in Azure ML Studio: 
    + for model (goto models section, select model. click "Publish"); 
    + for pipeline (goto Azure ML Designer and click "Publish" button for it)

* You can use TabularDataset to work with a structured (tabular) dataset in a Python script, without the need to register it in Azure ML workspace

* Azure ML Designer is only available with sku="enterprise" (not sku="basic") 

* Azure Stack Edge is hardware-as-a-service (HAAS) platform offered by Microsoft. Can be used to run IoT Edge to collect sensor info for machine learning purposes

* We use Azure ML SDK to create experiments that will process sparse data. To stop automatic featurization (cannot do that for sparse data) we need to turn off featurization attribute in <i>automl_settings</i> 
```
        automl_settings = {
            "n_cross_validations": 3,
            "primary_metric": 'r2_score',
            "enable_early_stopping": True,
            "experiment_timeout_hours": 1.0,
            "max_concurrent_iterations": 4,
            "max_cores_per_iteration": -1,
            "verbosity": logging.INFO,
            "featurization": "off"
        }

        automl_config = AutoMLConfig(task = 'regression',
                                    compute_target = compute_target,
                                    training_data = train_data,
                                    label_column_name = label,
                                    **automl_settings
                                    )

        ws = Workspace.from_config()
        experiment = Experiment(ws, "your-experiment-name")
        run = experiment.submit(automl_config, show_output=True)
```
* With the code below
    + DataDriftDetector will send an email if drift_coefficient = 0.4 when evaluated
    + after the configuration, the script will run drift detection on the day the script is executed
    + because <i>create_compute_target</i> parameter is set to True, it creates a new compute cluster for execution

```
        from azureml.core.import Experiment, Run, RunDetails
        from azureml.datadrift import DataDriftDetector, AlertConfiguration

        alert_config = AlertConfiguration('tonycecotto@yahoo.ca')

        datadrift = DataDriftDetector.create(ws,
                                            model.name,
                                            model.version,
                                            services,
                                            frequency="Day",
                                            alert_config=alert_config,
                                            drift_threshold=0.3)

        target_date = datetime.today()
        run = datadrift.run(target_date,services,feature_list=feature_list,create_compute_target=True)
```
* You can assign Workspace custom roles in the following 3 ways:
    + Execute the <i>az role assignment create</i> CLI command
    + Assign the role from the portal using Access Control (IAM) screen for the workspace
    + Execute the <i>az ml workspace share</i> CLI command
    <u>Note:</u> You cannot use <i>az ml workspace update</i> CLI command (not including role assignment)

* Various Compute Targets for various activities:
    + **Azure ML Compute Clusters**: <u>Training & inferencing for dev/test purposes</u>
        Compute Clusters are scalable machine learning platforms consisting of one or more CPU or GPU nodes. Compute clusters can scale from 0 to hundreds of nodes, depending on workload. Compute clusters support the use of low-priority VMs, whihc do not have giaranteed availability. Using low-priority VMs can help reduce machine learning costs. Azure ML compute clusters can be used for training pipelines, but <i>not for pipeline deployment because this functionality is not supported</i>
    + **Azure ML Compute Instance**: <u>Production grade training only</u>
        A compute instance is a single Azure-homed visrtual machine (VM) Azure ML compute instances are highly scalable cloud compute resources, whihc support multiple CPUs and large amounts of RAM based on VM size you select at deployment. Unlike a computer cluster, a compute instance cannot scale down to 0, meaning that usage charges accrue unless you power off the VM. Azure ML compute instances can be used for training pipelines, but <i>not for pipeline deployment because this functionality is not supported</i>
    + **AKS Clusters**: <u>Production grade deployment</u>
        AKS clusters are designed for heavy, real-time production workloads. One of the primary benefits of deploying to AKS is support for auto-scaling. This means that as workload increases or decreases, an AKS cluster can add or terminate cluster nodes. In addition for supporting multiple-node clusters, AKS can be used for experiments that require hardware acceleration via GPU or Field-Programmable Gate Arrays (FPGA)

* Open Neural Netwoek Exchange (ONNX) model accepts only Decision Tree and Random Forest for regression problems

* Use **Apache Ambari** for monitoring a Linux-based Azure HDInsight solution that is used to analyze large volumes of data. Apache Ambari simplifies the management and monitoring of Hadoop clusters by providing an easy-to-use Web UI bacled by its REST APIs. Apache Ambari is provided by default with a Linuz-based HDInsight cluster, therefore ensuring reduced configuration effort.
    + Azure Log Analytics is suited for querying diagnostic logs for resources created in Azure. 
    + Azure Sentinel is a cloud <i>security information event management (SIEM)</i> whihc cannot provide insights into the performance metrics within a cluster
    + HDInsight .NET SDK allows creation of code ie. logging framework with high resourcing costs

* **Azure HDInsight** allows you to use Apache Spark to train your models. Azure HDInsight provides a pre-configured environment with Apache Spark

* I used Azure ML Designer to create an inference pipeline and train a predictive model.How can I deploy the pipeline as a web service that can autoscale based on workload? 
    + Convert the training pipeline into a real-time inference pipeline
    + Create an AKS cluster
    + Deploy a real-time endpoint
        Inference is also known as <i>model scoring</i> Such models are trained on a datset and can then analyze data in real-time to provide predictions. Once your pipeline has trained a model. you convert the training pipeline into a real-time inference pipeline. This adds the supporting Web Service Input and Web Service Output modules to your pipeline
        An endpoint is the port-to-service mapping that is created when you deploy a web service. As part of deploying a real-time endpoint, you are required to specify a compute target, and publishing an autoscaling inference pipeline is only supported on AKS inference clusters. If you did nt create an AKS cluster prior to this step, you will need to define one before you can complete the deployment. Once the real-time endpoint has been deployed, apps and services can access the endpoint as they would any other REST API
        You should not set the pipeline as default for the dnpoint. Every endpoint has one default piepline. When you publish a new pipeline under an existinh endpoint, you can choose to make it default pipeline for that endpoint

* Use Azure ML SDK to generate feature importance. To determine whihc featues have the largest impact on a model's perdictions, use an interpretability technique that calculates and tracks feature importance. Azure ML supports the Permutations Feature Importance Explainer (PFI) for this purpose. PFI randomly shuffles features during model training, and then calculates the impact on model's performance.
    + Do not use a SHAP tree explainer for calculate feature importance. SHAP is not model-agniostic and is used for TreeBased models. 
    + Do not use a global surrogate with a mimic explainer to explain the model. A global surrogate is meant to be an interpretable approximation of a black box model. Black box models are those for whihc no explanation exists. Once  surrogate model is trained, mimic explainer can be used to interpret the model.

* You configure an Azure ML experiment using the following code:
```
        import azureml.core
        from azurem;.core import Workspace,Experiment
        ws = Workspace.from_config()
        script_params = ["--experiment_output",experiment_output]
        exp = Experiment(ws,experiment_name)
```
Too ensure output files are uploaded in real time you need to add <i>experiment_output = s.path.join(os.curdir,"logs") line of code.Azure ML supports a few locations for storing exepriment output. Files can be saved to storage on the local compute instance, but these files do not persist across the runs. To store files for later analysis and review, you either use Azure ML datastore, or write to the <i>outputs</i> or <i>logs</i> folders. However, for <i>outputs</i> folder, the files are not uploaded in real-time.

* You use Azure ML to create an ML pipeline. Need to ensure that files can be passed between pipelines steps using a named datastore. To do that:
    + regsiter a new Azure Storage file container daatstore
    + create a PipelineData object. Specify a name and output datastore
    + specify a PipelineData object for data output (in between steps)

* Questions / To Clarify
    + Difference between "Tabular Dataset" and "Pandas Dataset" (explain why the function below is correct!)
```
        jan_ds = time_series_ds.time_between(
                                    start_time=datetime(2019,12,31),
                                    end_time=datetime(2020,2,31),
                                    include_boundary=False<br/>
        )
        jan_ds.take(100).to_pandas_dataframe()
```
* More questions...
    + Azure Log Analytics - is it part of Azure Monitor? What about Azure App Insights?
    + Highly imbalanced datasets and right metrics
    + MLFlow - details
    + Difference between "publish to web service" and "publish to endpoint" when publishing a pipeline
    + What is "spearman correlation"

* The model deployments created by Azure Machine Learning can be configured to use one of two authentication methods:

    + **key-based**: A static key is used to authenticate to the web service.
        Web-services deployed on Azure Kubernetes Service (AKS) have key-based auth enabled by default.
        Azure Container Instances (ACI) deployed services have key-based auth disabled by default, but you can enable it by setting auth_enabled=Truewhen creating the ACI web-service

    + **token-based**: A temporary token must be obtained from the Azure Machine Learning workspace (using Azure Active Directory) and used to authenticate to the web service. This token expires after a period of time, and must be refreshed to continue working with the web service.
        When you enable token authentication for a web service, users must present an Azure Machine Learning JSON Web Token to the web service to access it. The token expires after a specified time-frame and needs to be refreshed to continue making calls.
        + Token authentication is disabled by default when you deploy to Azure Kubernetes Service.
        + Token authentication isn't supported when you deploy to Azure Container Instances.
        + Token authentication can't be used at the same time as key-based authentication.

* **Create Image Labeling Project** at https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-image-labeling-projects
* **Create Batch Inference Pipeline** at https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/machine-learning-pipelines/parallel-run/tabular-dataset-inference-iris.ipynb

* **Azure ML Fundamental Concepts
    + **Workspace** https://docs.microsoft.com/en-ca/azure/machine-learning/concept-workspace
    + **Environment** 

* Machine Learning Designer runs only on ML Compute Cluster
    + A web service deployed using Azure ML Designer requires by default configuration an authentication header to be passed in the headers for the requests.post method

* Training & Deployment on various compute targets
    + Azure ML Cluster Compute supports training only, because of 0 cluster nodes and low priority
    + Azure ML Compute Instance supports training only
    + AKS clusters support deployment only

* Add new kernels
    1. **conda create --name newenv** (create new envionment)
    2. **conda activate newenv** (activate environment)
    3. **conda install pip**
    4. **conda install ipykernel**
    5. **python -m ipykernel install --user --name newenv --display-name "Python (newenv)"** (create a kernel for conda env)

* TODO ---> Difference between "Interpretability" and "Explainability": https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-interpretability
* Example MLflow project at https://github.com/mlflow/mlflow-example

<hr/>

# Azure ML Study Material

## Models
* [KB: Python scikit-learn Linear Models] https://scikit-learn.org/stable/modules/linear_model.html
* [KB: Create an image labeling project and export labels] https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-image-labeling-projects
* [KB: Datastores And Datasets In Azure] https://k21academy.com/microsoft-azure/dp-100/datastores-and-datasets-in-azure/
* [KB: Set up AutoML to train a time-series forecasting model with Python] https://docs.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-forecast
* [KB: Data Cleaning Challenge: Scale and Normalize Data] https://www.kaggle.com/code/rtatman/data-cleaning-challenge-scale-and-normalize-data/notebook

## Computes
* [KB: Create compute targets for model training and deployment in Azure Machine Learning studio] https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-studio#ssh-access

## Comprehensive Azure ML Pipelines
* [Code: azureml examples] https://github.com/azure/azureml-examples
* [Code: AutoML examples] https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/automated-machine-learning/continuous-retraining/auto-ml-continuous-retraining.ipynb
* [KB: Introduction to pipelines] https://docs.microsoft.com/en-us/learn/modules/create-pipelines-in-aml/2-pipelines?WT.mc_id=esi_studyguide_content_wwl
* [KB: Create and run machine learning pipelines with Azure Machine Learning SDK] https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-machine-learning-pipelines?WT.mc_id=esi_studyguide_content_wwl
* [Tutorial: Build an Azure Machine Learning pipeline for image classification] https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-pipeline-python-sdk
* [<u>KB: Use automated ML in an Azure Machine Learning pipeline in Python</u>] https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-automlstep-in-pipelines
* [KB: What are Azure Machine Learning pipelines?] https://docs.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines#which-azure-pipeline-technology-should-i-use
* [KB: What are compute targets in Azure Machine Learning?] https://docs.microsoft.com/en-us/azure/machine-learning/concept-compute-target#train
* [KB: Introduction to private Docker container registries in Azure] https://docs.microsoft.com/en-us/azure/container-registry/container-registry-intro
* [KB:Moving data into and between ML pipeline steps (Python)] https://docs.microsoft.com/en-us/azure/machine-learning/how-to-move-data-in-out-of-pipelines?WT.mc_id=esi_studyguide_content_wwl
* [Tutorial: Orchestrate machine learning with pipelines] https://docs.microsoft.com/en-us/learn/modules/create-pipelines-in-aml/1-introduction
* [Tutorial: Automate model selection with Azure Automl] https://docs.microsoft.com/en-us/learn/modules/automate-model-selection-with-azure-automl/1-introduction
* [KB: Set up no-code AutoML training with the studio UI] https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-automated-ml-for-ml-models?WT.mc_id=esi_studyguide_content_wwl
* [KB: Configure authentication for models deployed as web services] https://docs.microsoft.com/en-us/azure/machine-learning/how-to-authenticate-web-service
* [KB: Build an Azure Machine Learning pipeline for image classification] https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-pipeline-python-sdk?WT.mc_id=esi_studyguide_content_wwl
* [KB: PipelineParameter Class] https://docs.microsoft.com/en-us/python/api/azureml-pipeline-core/azureml.pipeline.core.graph.pipelineparameter?view=azure-ml-py
* [KB: PYTORCH Creating batch endpoints in Azure ML] https://techcommunity.microsoft.com/t5/ai-machine-learning-blog/creating-batch-endpoints-in-azure-ml/ba-p/3039821
* [KB: Binary Classification Income Prediction] https://github.com/Azure/MachineLearningDesigner/blob/master/articles/samples/binary-classification-feature-selection-income-prediction.md

## Use HyperDrive to tune hyperparameters
* [KB: Hyperparameter tuning a model with Azure Machine Learning] https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters
* [KB: Tune hyperparameters with Azure Machine Learning] https://docs.microsoft.com/en-us/learn/modules/tune-hyperparameters-with-azure-machine-learning/1-introduction
* [Code: train-hyperparameter-tune-deploy-with-pytorch] https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/ml-frameworks/pytorch/train-hyperparameter-tune-deploy-with-pytorch

## Feature Selection
* [KB: <u>Filter Based Feature Selection</u>] https://docs.microsoft.com/en-us/azure/machine-learning/component-reference/filter-based-feature-selection
* [KB: <u>Permutation Feature Importance</u>] https://docs.microsoft.com/en-us/azure/machine-learning/component-reference/permutation-feature-importance
* [Whitepaper: SMOTE: Synthetic Minority Over-sampling Technique] https://jair.org/index.php/jair/article/view/10302

## Assess and mitigate model fairness
* [Code: Binary Classification with the UCI Creditcard Default Dataset] https://github.com/fairlearn/fairlearn/blob/main/notebooks/Binary%20Classification%20with%20the%20UCI%20Credit-card%20Default%20Dataset.ipynb
* [Code: Unfairness Mitigation with Fairlearn and Azure Machine Learning] https://github.com/Azure/MachineLearningNotebooks/blob/master/contrib/fairness/fairlearn-azureml-mitigation.ipynb
* [KB: Machine learning fairness] https://docs.microsoft.com/en-us/azure/machine-learning/concept-fairness-ml?WT.mc_id=esi_studyguide_content_wwl
* [KB: Use Azure Machine Learning with the Fairlearn open-source package to assess the fairness of ML models] https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-fairness-aml?WT.mc_id=esi_studyguide_content_wwl
* [Tutorial: Explore differential privacy] https://docs.microsoft.com/en-us/learn/modules/explore-differential-privacy/?WT.mc_id=esi_studyguide_content_wwl

## Deployments
* [KB: Deploy machine learning models to Azure] https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-and-where?tabs=python#registermodel?WT.mc_id=esi_studyguide_content_wwl
* [KB: Monitor Azure resources with Azure Monitor] https://docs.microsoft.com/en-us/azure/azure-monitor/essentials/monitor-azure-resource
* [KB: Enterprise security and governance for Azure Machine Learning] https://docs.microsoft.com/en-us/azure/machine-learning/concept-enterprise-security?WT.mc_id=esi_studyguide_content_wwl
* [KB: Deploy a model to an Azure Kubernetes Service cluster] https://docs.microsoft.com/en-us/azure/machine-learning/how-to-deploy-azure-kubernetes-service?tabs=python
* [KB: Designer - deploy a machine learning model] https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-designer-automobile-price-deploy?WT.mc_id=esi_studyguide_content_wwl
    + Create a real-time inference pipeline
    + Create an inferencing cluster
    + Deploy the real-time endpoints
    + Test the real-time endpoints 
* [KB: Configure authentication for models deployed as web services] https://docs.microsoft.com/en-us/azure/machine-learning/how-to-authenticate-web-service
* [KB: Manage HDInsight clusters by using the Apache Ambari REST API] https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-hadoop-manage-ambari-rest-api
* [KB: Enterprise security and governance for Azure Machine Learning] https://docs.microsoft.com/en-us/azure/machine-learning/concept-enterprise-security?WT.mc_id=esi_studyguide_content_wwl

## Containers
* [KB: Configure a custom container for Azure App Service] https://docs.microsoft.com/en-us/azure/app-service/configure-custom-container?pivots=container-linux

## Security 
* [KB: How to use Managed Identities in Azure Machine Learning] https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-managed-identities?tabs=python
* [KB: Create or update Azure custom roles using the Azure portal] https://docs.microsoft.com/en-us/azure/role-based-access-control/custom-roles-portal
* [KB: Understand Azure role definitions] https://docs.microsoft.com/en-us/azure/role-based-access-control/role-definitions#control-and-data-actions

# Check It Out!
* [Github: Intelligent Cloud Demonstrations] https://github.com/dxuk/Stand-Demos
* [KB: Run batch predictions using Azure Machine Learning designer] https://docs.microsoft.com/en-us/azure/machine-learning/how-to-run-batch-predictions-designer 
* [Notebook: Tabular Time Series Related API Demo with NOAA Weather Data] https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/work-with-data/datasets-tutorial/timeseries-datasets/tabular-timeseries-dataset-filtering.ipynb
* [SUMMARY: Build and operate machine learning solutions with Azure Machine Learning] https://docs.microsoft.com/en-us/learn/paths/build-ai-solutions-with-azure-ml-service/ 

<hr/>

## Azure ML Links: 
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
16. https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace-cli?tabs=vnetpleconfigurationsv1cli%2Ccreatenewresources%2Cworkspaceupdatev1%2Cworkspacesynckeysv1%2Cworkspacedeletev1
17. https://docs.microsoft.com/en-us/cli/azure/ml/workspace?view=azure-cli-latest
18. https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train
19. https://docs.microsoft.com/en-us/azure/machine-learning/concept-fairness-ml
20. https://docs.microsoft.com/en-us/azure/machine-learning/how-to-assign-roles

## Statistics Links
1. Understanding Boxplots - https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51#:~:text=A%20boxplot%20is%20a%20standardized,and%20what%20their%20values%20are.
2. Precision vs Recall - https://medium.com/@shrutisaxena0617/precision-vs-recall-386cf9f89488#:~:text=Precision%20and%20recall%20are%20two,correctly%20classified%20by%20your%20algorithm.
