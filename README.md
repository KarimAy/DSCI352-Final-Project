README – Telco Churn & Bitcoin LSTM Forecasting Project
USC DSCI 352: Applied Machine Learning & Data Mining
Final Project – Fall 2025
Authors: Karim Ayoub, Jaclyn Chung, Mikhael Hilaly

PROJECT OVERVIEW

This project includes two major machine learning workflows:

Telco Customer Churn Prediction (structured tabular data)

Models trained: Logistic Regression, Random Forest, Gradient Boosting, SGDClassifier, Keras MLP.

Full preprocessing pipeline using imputation, scaling, and OneHotEncoder.

Gradient tracking for deep learning model analysis.

Model comparison using AUC and accuracy.

Deployment-style local inference pipeline using a serialized SGDClassifier.

Lightweight artifact (telco_artifact.pkl) suitable for AWS-style deployment.

Bitcoin Price Forecasting (time-series deep learning)

Multivariate LSTM forecasting using hourly BTC-USD data.

Sliding-window dataset creation.

Feature engineering (returns, rolling averages, volatility, momentum).

MinMax scaling.

Model evaluation using RMSE, MAE, and directional accuracy.

All required deliverables are included, including training scripts, prediction scripts, and saved artifacts.

PROJECT STRUCTURE

DSCI352-Final-Project/
telco_training.py or aws_code.py (main Telco training script)
predict_telco.py (local prediction tool)
sgd_pipeline.pkl (saved sklearn pipeline for inference)
telco_artifact.pkl (lightweight AWS-ready artifact)
keras_gradients.json (gradient statistics during Keras training)
model_leaderboard_telco.json (model performance summary)
new_customers.csv (example input)
predictions.csv (example output)
BTC-LSTM/ (Bitcoin LSTM notebook and data)

INSTALLING DEPENDENCIES

Use Python 3.10 or later.

(Recommended) Create a virtual environment:

conda create -n dsci352 python=3.10
conda activate dsci352

Install requirements:

pip install pandas numpy scikit-learn tensorflow joblib boto3

(If a requirements.txt file is provided, run pip install -r requirements.txt.)

HOW TO RUN THE TELCO TRAINING PIPELINE

This script trains all sklearn and Keras models, evaluates their performance, tracks gradients, and saves all project artifacts.

Run:

python aws_code.py

This will:
- Train Logistic Regression, Random Forest, Gradient Boosting, SGDClassifier.
- Train a Keras MLP and track gradient statistics.
- Evaluate AUC and accuracy for all models.
- Save:
keras_gradients.json
model_leaderboard_telco.json
sgd_pipeline.pkl
telco_artifact.pkl

You should see console output showing:
- Training logs
- AUC results
- Confirmation that each artifact has been saved

HOW TO RUN THE LOCAL PREDICTION PIPELINE

After training, you can generate churn predictions for new customers.

Run:

python predict_telco.py --input new_customers.csv --output predictions.csv

Expected console output:
Loading trained pipeline (sgd_pipeline.pkl)...
Loading new_customers.csv...
Generating churn probabilities...
Saved predictions → predictions.csv

The output file contains:
customer_index
churn_probability
predicted_label

AWS-LIGHT ARTIFACT (OPTIONAL)

The file telco_artifact.pkl contains:
- Preprocessing configuration
- Best sklearn model
- Numeric and categorical feature lists

This artifact is structured so it can be adapted for AWS Lambda or other deployment environments.

To enable S3 uploading in the script:
Set ENABLE_AWS_EXPORT = True in the training script.
Set the following environment variables:
MODEL_BUCKET
MODEL_KEY

Example:
export ENABLE_AWS_EXPORT=True
export MODEL_BUCKET=your-bucket
export MODEL_KEY=models/telco_churn_light.json

HOW TO RUN THE BITCOIN LSTM MODEL

Navigate to BTC-LSTM folder and open the Jupyter notebook:

jupyter notebook BTC-LSTM/btc_lstm_training.ipynb

The notebook contains:
- Data preprocessing
- LSTM architecture
- Training and evaluation
- Visualizations
- RMSE, MAE, and directional accuracy metrics

OUTPUT FILES AND ARTIFACTS

keras_gradients.json
Gradient metrics for every epoch of Keras training.

model_leaderboard_telco.json
Performance of all sklearn and Keras models.

sgd_pipeline.pkl
Serialized sklearn pipeline + SGDClassifier for local inference.

telco_artifact.pkl
Lightweight deployment artifact (preprocessor, model, feature mappings).

predictions.csv
Output from running predict_telco.py.

REPRODUCIBILITY NOTES

Random seeds are fixed (random_state=42) for all sklearn operations.

Telco dataset WA_Fn-UseC_-Telco-Customer-Churn.csv must be in project root.

TensorFlow outputs may vary slightly depending on CPU instructions.
