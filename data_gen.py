# %%
import pandas as pd
import numpy as np
# %%
def create_cc_persons():
    demograpgics = pd.read_csv('datasets/exam-1/demographics.csv')
    cc = pd.read_csv('datasets/exam-1/cc.csv')
    cc.sort_values(by=['cc_no', 'pos_dt'], inplace=True)
    cc_id = pd.merge(cc, demograpgics[['id','cc_no']], on='cc_no')
    cc_id['count'] = 1
    grouped = cc_id.groupby(['id', 'pos_dt'])
    cc_persons = grouped.sum().reset_index()
    origin_timestamp = pd.Timestamp(cc['pos_dt'].min())
    cc_persons['pos_dt_index'] = cc_persons.apply(lambda row: (pd.Timestamp(row['pos_dt']) - origin_timestamp).days, axis=1)
    cc_persons.to_csv('datasets/exam-1/cc_persons.csv')

def create_weekly_kplus(kplus, padding_value):
    sunday_id_hash = {sunday: i for i, sunday in enumerate(
        kplus.dropna().groupby('sunday', sort=True).groups.keys())}
    max_len = len(sunday_id_hash)

    def create_sequence(group, padding_value):
        
        if not group.isna().values.any():
            origin = sunday_id_hash[group.iloc[0]['sunday']]
            seq = group[['kp_txn_amt', 'kp_txn_count']].to_numpy()
            pre_padding = np.ones((origin, 2), dtype=np.float32) * padding_value
            post_padding = np.ones((max_len - origin - len(seq), 2), dtype=np.float32) * padding_value
            seq = np.concatenate((pre_padding, seq, post_padding), axis=0)
        else:
            seq = np.ones((max_len, 2), dtype=np.float32) * padding_value
        return seq
    xs = np.array([create_sequence(group, padding_value)
                   for _, group in kplus.groupby('id')])
    return xs

def create_credit_seq(cc_persons, padding_value):
    xs = [] # [[cc_txn_amt, count], ...]

    origin_timestamp = pd.Timestamp(cc_persons['pos_dt'].dropna().min())
    max_len_seq = (pd.Timestamp(cc_persons['pos_dt'].dropna().max()) - origin_timestamp).days + 1
    
    empty_seq = np.ones((max_len_seq, 2), dtype=np.float32) * padding_value
    for _id, group in cc_persons.groupby('id'):
        seq = empty_seq.copy()
        if not group.isna().values.any():
            seq[group['pos_dt_index'].to_numpy(int)] = group[['cc_txn_amt', 'count']].to_numpy()
        xs.append(seq)
    xs = np.asarray(xs)
    return xs
#%%
def create_daily_kplus(kplus, padding_value):
    start_timestamp = pd.Timestamp('2018-01-01')
    end_timestamp = pd.Timestamp('2018-06-30')

    max_len = (end_timestamp - start_timestamp).days + 1
    xs = []
    for _, group in kplus.groupby('id'):
        seq = np.ones((max_len, 2), dtype=np.float32) * padding_value
        if not group.isna().values.any():
            sunday_indexes = group.apply(lambda row: (pd.Timestamp(row['sunday']) - start_timestamp).days, axis=1).to_numpy(int)
            seq[sunday_indexes] = group[['kp_txn_amt', 'kp_txn_count']].to_numpy()
        xs.append(seq)
    xs = np.asarray(xs)
    return xs
# %%
print('load data gen')
