#!/usr/bin/env python
# coding: utf-8

# In[1]:




# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


# In[1]:




import keras.backend as K
import numpy as np
import pandas as pd
from keras.models import Model, Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from tqdm import tnrange, tqdm, trange

from callbacks import EvaluateSMAPE
from data_gen import (create_credit_seq, create_daily_kplus,
                      create_demographic_data, create_weekly_credit,
                      create_weekly_kplus, deimbalance_transaction_datagen,
                      ensemble_ocp_demo_datagen,
                      create_cc_weekly,
                      create_raw_demographic)
from losses import SMAPE_loss
from util import modified_SMAPE


# In[2]:




demograpgics = pd.read_csv('datasets/exam-1/demographics.csv')
cc = pd.read_csv('datasets/exam-1/cc.csv')
cc.sort_values(by=['cc_no', 'pos_dt'], inplace=True)
kplus = pd.read_csv('datasets/exam-1/kplus.csv')
kplus.sort_values(by=['id', 'sunday'], inplace=True)
train_set = pd.read_csv('datasets/exam-1/train.csv')
test_set = pd.read_csv('datasets/exam-1/test.csv')
raw_demographics = create_raw_demographic(demograpgics)

scaled_kplus = kplus.copy()
kplus_scaler = StandardScaler()
scaled_kplus[['kp_txn_amt', 'kp_txn_count']] = kplus_scaler.fit_transform(
    kplus[['kp_txn_amt', 'kp_txn_count']])

# cc_weekly = pd.read_csv('datasets/exam-1/cc_weekly_persons.csv')
cc_weekly = create_cc_weekly(cc, demograpgics)
cc_weekly_scaler = StandardScaler()
cc_weekly[['cc_txn_amt', 'count']] = cc_weekly_scaler.fit_transform(
    cc_weekly[['cc_txn_amt', 'count']])

train_scaler = StandardScaler()
scaled_train_set = train_set.copy()
scaled_train_set.set_index('id', inplace=True)
scaled_train_set['income'] = train_scaler.fit_transform(
    np.expand_dims(scaled_train_set['income'], axis=2))


# In[7]:




def create_all_x_data(ids):
    ids_df = pd.DataFrame(ids, columns=['id'])
    kplus = pd.merge(scaled_kplus, ids_df, on='id', how='right')
    _cc_weekly = pd.merge(cc_weekly, ids_df, on='id', how='right')

    padding_value = float(-100)
    kplus_padding = cc_padding = padding_value
    kplus_padding = kplus_scaler.transform([[0, 0]])
    cc_weekly_padding = cc_weekly_scaler.transform([[0, 0]])

    kplus_xs = create_weekly_kplus(kplus, kplus_padding)
    gap_padding = np.ones((len(kplus_xs), 1, 2),
                          dtype=np.float32) * kplus_padding
    kplus_xs = np.concatenate((kplus_xs, gap_padding,), axis=1)
    cc_weekly_xs = create_weekly_credit(_cc_weekly, cc_weekly_padding)
    transaction_xs = np.concatenate((kplus_xs, cc_weekly_xs), axis=2)

    demographic_xs = create_demographic_data(
        raw_demographics.reset_index().set_index('id').loc[ids])
    return demographic_xs, transaction_xs


# In[4]:




model_path = 'weight/demo_kplus_cc-weekly_10/ep50-val_SMAPE:92.13181111812591-test91.74.h5'
model = load_model(model_path, custom_objects={
                   'loss_func': SMAPE_loss(train_scaler)})


# In[8]:


demo_test_xs, transaction_test_xs = create_all_x_data(
    test_set['id'].to_numpy())


# In[12]:


ys_pred = model.predict([demo_test_xs, transaction_test_xs], verbose=1)
test_incomes = train_scaler.inverse_transform(ys_pred)


# In[ ]:




output_path = 'O_0055.csv'
answer = test_set.copy()
answer['income'] = test_incomes
answer.to_csv(output_path)


# In[1]:




