# RESEARCH
Model Training and Deployment Workflow

This document outlines the standard workflow for configuring, training, and deploying the model.

Workflow Steps

Dataset Processing
Ensure that your dataset is fully processed, cleaned, and ready for training.

Hyperparameter Configuration
Adjust the training hyperparameters to suit your specific run.

Update general hyperparameters in the train_config file.

Adjust the batch_size directly within the data module.

Learning Rate Setup
By default, the learning rate is split into two distinct parameters: one for the classification/output head and one for the CNN backbone.

Note: You are encouraged to experiment with these. You can try combining them into a single learning rate or test different scheduling and configuration strategies.

Model Training
Once configuration is complete, initiate the model training process.

Deployment
After the model has finished training and evaluation, it is ready to be deployed to your target environment.
