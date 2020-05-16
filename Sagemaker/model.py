#!/usr/bin/env python
# coding: utf-8

# <h1>SMS Spam Classifier</h1>
# <br />
# This notebook shows how to implement a basic spam classifier for SMS messages using Amazon SageMaker built-in linear learner algorithm.
# The idea is to use the SMS spam collection dataset available at <a href="https://archive.ics.uci.edu/ml/datasets/sms+spam+collection">https://archive.ics.uci.edu/ml/datasets/sms+spam+collection</a> to train and deploy a binary classification model by leveraging on the built-in Linear Learner algoirithm available in Amazon SageMaker.
# 
# Amazon SageMaker's Linear Learner algorithm extends upon typical linear models by training many models in parallel, in a computationally efficient manner. Each model has a different set of hyperparameters, and then the algorithm finds the set that optimizes a specific criteria. This can provide substantially more accurate models than typical linear algorithms at the same, or lower, cost.

# Let's get started by setting some configuration variables and getting the Amazon SageMaker session and the current execution role, using the Amazon SageMaker high-level SDK for Python.

# In[1]:


from sagemaker import get_execution_role

bucket_name = 'email-spam-sagemaker'

role = get_execution_role()
bucket_key_prefix = 'sms-spam-classifier'
vocabulary_length = 9013

print(role)


# We now download the spam collection dataset, unzip it and read the first 10 rows.

# In[2]:


get_ipython().system('mkdir -p dataset')
get_ipython().system('curl https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip -o dataset/smsspamcollection.zip')
get_ipython().system('unzip -o dataset/smsspamcollection.zip -d dataset')
get_ipython().system('head -10 dataset/SMSSpamCollection')


# We now load the dataset into a Pandas dataframe and execute some data preparation.
# More specifically we have to:
# <ul>
#     <li>replace the target column values (ham/spam) with numeric values (0/1)</li>
#     <li>tokenize the sms messages and encode based on word counts</li>
#     <li>split into train and test sets</li>
#     <li>upload to a S3 bucket for training</li>
# </ul>

# In[3]:


import pandas as pd
import numpy as np
import pickle
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences

df = pd.read_csv('dataset/SMSSpamCollection', sep='\t', header=None)
df[df.columns[0]] = df[df.columns[0]].map({'ham': 0, 'spam': 1})

targets = df[df.columns[0]].values
messages = df[df.columns[1]].values

# one hot encoding for each SMS message
one_hot_data = one_hot_encode(messages, vocabulary_length)
encoded_messages = vectorize_sequences(one_hot_data, vocabulary_length)

df2 = pd.DataFrame(encoded_messages)
df2.insert(0, 'spam', targets)

# Split into training and validation sets (80%/20% split)
split_index = int(np.ceil(df.shape[0] * 0.8))
train_set = df2[:split_index]
val_set = df2[split_index:]

train_set.to_csv('dataset/sms_train_set.csv', header=False, index=False)
val_set.to_csv('dataset/sms_val_set.csv', header=False, index=False)


# We have to upload the two files back to Amazon S3 in order to be accessed by the Amazon SageMaker training cluster.

# In[4]:


import boto3

s3 = boto3.resource('s3')
target_bucket = s3.Bucket(bucket_name)

with open('dataset/sms_train_set.csv', 'rb') as data:
    target_bucket.upload_fileobj(data, '{0}/train/sms_train_set.csv'.format(bucket_key_prefix))
    
with open('dataset/sms_val_set.csv', 'rb') as data:
    target_bucket.upload_fileobj(data, '{0}/val/sms_val_set.csv'.format(bucket_key_prefix))


# <h2>Training the model with Linear Learner</h2>
# 
# We are now ready to run the training using the Amazon SageMaker Linear Learner built-in algorithm. First let's get the linear larner container.

# In[5]:


import boto3

from sagemaker.amazon.amazon_estimator import get_image_uri
container = get_image_uri(boto3.Session().region_name, 'linear-learner', repo_version="latest")


# Next we'll kick off the base estimator, making sure to pass in the necessary hyperparameters. Notice:
# 
# <ul>
#     <li>feature_dim is set to the same dimension of the vocabulary.</li>
# <li>predictor_type is set to 'binary_classifier' since we are trying to predict whether a SMS message is spam or not.</li>
# <li>mini_batch_size is set to 100.</li>
# <ul>

# In[6]:


import sagemaker

output_path = 's3://{0}/{1}/output'.format(bucket_name, bucket_key_prefix)

linear = sagemaker.estimator.Estimator(container,
                                       role, 
                                       train_instance_count=1, 
                                       train_instance_type='ml.c5.2xlarge',
                                       output_path=output_path,
                                       base_job_name='sms-spam-classifier-ll')
linear.set_hyperparameters(feature_dim=vocabulary_length,
                           predictor_type='binary_classifier',
                           mini_batch_size=100)

train_config = sagemaker.session.s3_input('s3://{0}/{1}/train/{2}'
                                          .format(bucket_name, bucket_key_prefix, 'sms_train_set.csv'), 
                                          content_type='text/csv')
test_config = sagemaker.session.s3_input('s3://{0}/{1}/val/{2}'
                                         .format(bucket_name, bucket_key_prefix, 'sms_val_set.csv'), 
                                         content_type='text/csv')

linear.fit({'train': train_config, 'test': test_config })


# <h3><span style="color:red">THE FOLLOWING STEPS ARE NOT MANDATORY IF YOU PLAN TO DEPLOY TO AWS LAMBDA AND ARE INCLUDED IN THIS NOTEBOOK FOR EDUCATIONAL PURPOSES.</span></h3>

# <h2>Deploying the model</h2>
# 
# Let's deploy the trained model to a real-time inference endpoint fully-managed by Amazon SageMaker.

# In[7]:


pred = linear.deploy(initial_instance_count=1,
                     instance_type='ml.m5.large')


# <h2>Executing Inferences</h2>
# 
# Now, we can invoke the Amazon SageMaker real-time endpoint to execute some inferences, by providing SMS messages and getting the predicted label (SPAM = 1, HAM = 0) and the related probability.

# In[8]:


from sagemaker.predictor import RealTimePredictor
from sms_spam_classifier_utilities import one_hot_encode
from sms_spam_classifier_utilities import vectorize_sequences
from sagemaker.predictor import csv_serializer, json_deserializer

# Uncomment the following line to connect to an existing endpoint.
# pred = RealTimePredictor('<endpoint_name>')

pred.content_type = 'text/csv'
pred.serializer = csv_serializer
pred.deserializer = json_deserializer

test_messages = ["FreeMsg: Txt: CALL to No: 86888 & claim your reward of 3 hours talk time to use from your phone now! ubscribe6GBP/ mnth inc 3hrs 16 stop?txtStop"]
one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)

result = pred.predict(encoded_test_messages)
print(result)


# <h2>Cleaning-up</h2>
# 
# When done, we can delete the Amazon SageMaker real-time inference endpoint.

# In[9]:


pred.delete_endpoint()


# In[ ]:




