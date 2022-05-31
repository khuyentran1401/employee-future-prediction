# Employee Future Prediction

## Motivation
In a data science team, it is common to continuously try to find a better model than the existing one in production. It is important to make sure the service doesn't break when the new model is deployed. 

This project demonstrates how to use DagsHub and GitHub Actions to:
- automatically test a pull request from a team member
- merge a pull request when all tests passed
- deploy the ML model to the existing service

![](https://cdn-images-1.medium.com/max/800/1*VZLOx6sCq9_Dj1-44mxKOQ.png)

Here is the summary of the workflow:

## Experiment on DagsHub
After experimenting with different parameters using [MLFlow](https://mlflow.org/) and [DagsHub](https://towardsdatascience.com/dagshub-a-github-supplement-for-data-scientists-and-ml-engineers-9ecaf49cc505), we choose a combinations of parameters that gives a better performance than the existing model in production and commit the code to Git. 

![](https://cdn-images-1.medium.com/max/800/1*AVtGMnz8_2K3dOtQAKCTdQ.png)

## Use GitHub Actions to Test Model
The first workflow named [test_model.yaml](https://github.com/khuyentran1401/employee-future-prediction/blob/master/.github/workflows/test_model.yaml) automatically tests a new pull request, which can only be merged if all tests are passed.

![](https://cdn-images-1.medium.com/max/800/1*Prnyik5wQ2A5ciZP2NmRhw.png)

## Use GitHub Actions to Deploy Model After Merging
The second workflow named [deploy_app.yaml](https://github.com/khuyentran1401/employee-future-prediction/blob/master/.github/workflows/deploy_app.yaml) automatically deploy the new model to the existing service after the pull request is merged.

![](https://cdn-images-1.medium.com/max/800/1*gb37ASDRRILsKJYe3CBFyw.png).
