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

### Last Question: 18 of 54<hr/>


# Links: 
1. https://christophm.github.io/interpretable-ml-book/ 
2. https://shap.readthedocs.io/en/latest/ 
3. https://christophm.github.io/interpretable-ml-book/shapley.html
4. https://github.com/MicrosoftLearning/DP100/blob/master/07B%20-%20Creating%20a%20Batch%20Inferencing%20Service.ipynb
5. https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets



