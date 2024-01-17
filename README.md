# Machine Learning Approaches for Text Classification: A Study on Spam Detection

## Introduction:

Welcome to the "Machine Learning Approaches for Text Classification: A Study on Spam Detection" project! In this venture, we aim to address the challenge of categorizing text into 'spam' or 'ham (not spam)' using various machine learning algorithms. Our focus is on establishing a robust training-validation-test framework and employing a systematic approach to hyperparameter tuning. The significance of spam detectors in our daily interactions with text messaging cannot be overstated, making the deployment of effective classifiers crucial. Leveraging the power of machine learning, this project explores the training and fine-tuning of diverse models tailored specifically for spam detection.

## Project Scope:

This project is a part of CMPUT 466, and our exploration spans a spectrum of diverse machine learning algorithms, satisfying a minimum of three stipulated models. Given the nature of the Classification problem, models like Linear Regression and Gaussian Mixture Models are not suitable, leading us to focus on Logistic Regression, Single-layer Neural Network, and Multilayer Neural Network. Alongside a comprehensive comparison of these advanced models, we establish trivial baselines, recognizing the importance of simplicity in our evaluation metrics.

## Problem Formulation:

The input data for our models consists of text messages, with binary output—'1' for spam and '0' for not spam. We sourced our dataset from [Kaggle's SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/), comprising approximately 5574 messages from diverse and openly available sources on the internet. The dataset amalgamates messages from repositories like Grumbletext, NUS SMS Corpus, Caroline Tag’s PhD thesis, and SMS Spam Corpus v.0.1.

## Approaches and Baselines:

### Feature Extraction:
The primary approach involves feature extraction using the Term Frequency-Inverse Document Frequency (TF-IDF) measure. TF-IDF dynamically gauges the significance of a word within a document collection, mitigating the impact of ubiquitous words like "it" and "the."

### Baseline Configurations:
For all three machine learning models, we utilized the scikit-learn library's default configurations as baselines. A systematic exploration of hyperparameter space was conducted through a grid search, complemented with cross-validation for robust performance evaluation.

### Tools and Packages:
Make sure you have the following tools and packages installed before running the code:
- Python (version >= 3.6)
- scikit-learn
- numpy
- pandas
- matplotlib
- tensorflow (for Neural Network models)

### Dataset Splitting:
The dataset was divided into training and testing sets, with 80% allocated for training and 20% for testing. This division ensures a comprehensive evaluation of the model's generalization performance on unseen data.

### Performance Metrics:
The execution time for the entire process, including hyperparameter tuning, training, and evaluation, was recorded. Key metrics, such as accuracy and best hyperparameters, serve as benchmarks for assessing model performance and configuration optimization.

## Results:

Detailed results, including hyperparameters for each approach, will be presented in the results section.

## Getting Started:

To replicate or build upon our findings, follow the steps outlined in the accompanying code repository. Ensure you have the required dependencies installed using the following command:

```bash
pip install -r requirements.txt
