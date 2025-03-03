# %%
import pandas as pd
import numpy as np
from keras.utils import to_categorical
import scipy.stats as st

# %%


def create_cc_persons():
    demograpgics = pd.read_csv('datasets/exam-1/demographics.csv')
    cc = pd.read_csv('datasets/exam-1/cc.csv')
    cc.sort_values(by=['cc_no', 'pos_dt'], inplace=True)
    cc_id = pd.merge(cc, demograpgics[['id', 'cc_no']], on='cc_no')
    cc_id['count'] = 1
    grouped = cc_id.groupby(['id', 'pos_dt'])
    cc_persons = grouped.sum().reset_index()
    origin_timestamp = pd.Timestamp(cc['pos_dt'].min())
    cc_persons['pos_dt_index'] = cc_persons.apply(lambda row: (
        pd.Timestamp(row['pos_dt']) - origin_timestamp).days, axis=1)
    cc_persons.to_csv('datasets/exam-1/cc_persons.csv')


def create_raw_demographic(demographics):
    # demographics = pd.read_csv('datasets/exam-1/demographics.csv')
    raw_demograhgics = demographics.drop('cc_no', axis=1).drop_duplicates()
    raw_demograhgics['ocp_cd'][raw_demograhgics['ocp_cd'].isna()] = 0
    raw_demograhgics.set_index('id', inplace=True)
    raw_demograhgics.to_csv('datasets/exam-1/raw_demograhgics.csv')
    return raw_demograhgics

def create_weekly_kplus(kplus, padding_value):
    # sunday_id_hash = {sunday: i for i, sunday in enumerate(
    #     kplus.dropna().groupby('sunday', sort=True).groups.keys())}
    # max_len = len(sunday_id_hash)
    max_len = 25
    def create_sequence(group, padding_value):

        if not group.isna().values.any():
            origin = pd.Timestamp(group.iloc[0]['sunday']).week - 1
            seq = group[['kp_txn_amt', 'kp_txn_count']].to_numpy()
            pre_padding = np.ones(
                (origin, 2), dtype=np.float32) * padding_value
            post_padding = np.ones(
                (max_len - origin - len(seq), 2), dtype=np.float32) * padding_value
            seq = np.concatenate((pre_padding, seq, post_padding), axis=0)
        else:
            seq = np.ones((max_len, 2), dtype=np.float32) * padding_value
        return seq
    xs = np.array([create_sequence(group, padding_value)
                   for _, group in kplus.groupby('id')])
    return xs


def create_credit_seq(cc_persons, padding_value):
    xs = []  # [[cc_txn_amt, count], ...]

    origin_timestamp = pd.Timestamp(cc_persons['pos_dt'].dropna().min())
    max_len_seq = (pd.Timestamp(
        cc_persons['pos_dt'].dropna().max()) - origin_timestamp).days + 1

    empty_seq = np.ones((max_len_seq, 2), dtype=np.float32) * padding_value
    for _id, group in cc_persons.groupby('id'):
        seq = empty_seq.copy()
        if not group.isna().values.any():
            seq[group['pos_dt_index'].to_numpy(
                int)] = group[['cc_txn_amt', 'count']].to_numpy()
        xs.append(seq)
    xs = np.asarray(xs)
    return xs


def create_weekly_credit(cc_weekly, padding_value):
    empty_seq = np.ones(
        (int(cc_weekly['pos_week_index'].max()) + 1, 2), dtype=np.float32) * padding_value
    seqs = []
    for _id, group in cc_weekly.groupby('id'):
        seq = empty_seq.copy()
        if not group.isna().values.any():
            seq[group['pos_week_index'].astype(np.int)] = group[[
                'cc_txn_amt', 'count']]
        seqs.append(seq)
    seqs = np.asarray(seqs)
    return seqs


def create_daily_kplus(kplus, padding_value):
    start_timestamp = pd.Timestamp('2018-01-01')
    end_timestamp = pd.Timestamp('2018-06-30')

    max_len = (end_timestamp - start_timestamp).days + 1
    xs = []
    for _, group in kplus.groupby('id'):
        seq = np.ones((max_len, 2), dtype=np.float32) * padding_value
        if not group.isna().values.any():
            sunday_indexes = group.apply(lambda row: (pd.Timestamp(
                row['sunday']) - start_timestamp).days, axis=1).to_numpy(int)
            seq[sunday_indexes] = group[['kp_txn_amt', 'kp_txn_count']].to_numpy()
        xs.append(seq)
    xs = np.asarray(xs)
    return xs


def convert_demographic_onehot(person):
    gender = to_categorical(person['gender'] - 1, 2)
    age = to_categorical(person['age'] - 2, 5)
    if not np.isnan(person['ocp_cd']):
        ocp_cd = to_categorical(person['ocp_cd'], 14)
    else:
        ocp_cd = to_categorical(0, 14)
    return np.concatenate((gender, age, ocp_cd), axis=0)


def create_demographic_data(raw_demographic):
    return np.array([convert_demographic_onehot(person) for _, person in raw_demographic.iterrows()])


# %%
def ensemble_ocp_demo_datagen(raw_demographics,  batch_size=32, abuntant_ratio=0.6):
    demographics = raw_demographics.set_index('ocp_cd')

    abuntant_ocp = [3, 9]
    rare_ocp = [1, 2, 4, 5, 6, 7, 8, 11, 12, 13]

    abuntant_demo = demographics.loc[abuntant_ocp]
    rare_demo = demographics.loc[rare_ocp]
    while True:
        n_abuntant_sample = int(batch_size*abuntant_ratio)
        demo = abuntant_demo.sample(n_abuntant_sample).append(
            rare_demo.sample(batch_size - n_abuntant_sample))
        demo = demo.sample(frac=1).reset_index()
        yield create_demographic_data(demo), demo['income'].to_numpy()


def deimbalance_income_ids_generater(train_set, batch_size, abuntant_batch_ratio=0.6, percentile=0.2):
    train_set = train_set.reset_index().copy()
    train_set['z'] = (train_set['income'] -
                      train_set['income'].mean()) / train_set['income'].std()
    abuntant_ids = train_set[(train_set['z'] >= st.norm.ppf(
        percentile)) & (train_set['z'] <= st.norm.ppf(1 - percentile))]['id'].to_numpy()
    rare_ids = train_set[(train_set['z'] < st.norm.ppf(
        percentile)) | (train_set['z'] > st.norm.ppf(1-percentile))]['id'].to_numpy()
    n_abuntant = int(batch_size*abuntant_batch_ratio)
    while True:
        ids = np.random.choice(abuntant_ids, n_abuntant, replace=False).tolist(
        ) + np.random.choice(rare_ids, batch_size-n_abuntant, replace=False).tolist()
        np.random.shuffle(ids)
        yield ids


def create_transaction_xs(kplus, cc_weekly, kplus_padding, cc_weekly_padding):
    kplus_xs = create_weekly_kplus(kplus, kplus_padding)
    gap_padding = np.ones((len(kplus_xs), 1, 2),
                          dtype=np.float32) * kplus_padding
    kplus_xs = np.concatenate((gap_padding, kplus_xs,), axis=1)
    cc_xs = create_weekly_credit(cc_weekly, cc_weekly_padding)
    xs = np.concatenate((kplus_xs, cc_xs), axis=2)
    return xs


def deimbalance_transaction_datagen(kplus, cc_weekly, train_set,
                                    kplus_padding, cc_weekly_padding,
                                    batch_size=32, abuntant_batch_ratio=0.6, percentile=0.2):
    ids_gen = deimbalance_income_ids_generater(
        train_set, batch_size, abuntant_batch_ratio, percentile)
    if kplus.index.name != 'id':
        kplus = kplus.set_index('id')
    if cc_weekly.index.name != 'id':
        cc_weekly = cc_weekly.set_index('id')
    if train_set.index.name != 'id':
        train_set = train_set.set_index('id')
    while True:
        ids = next(ids_gen)
        xs = create_transaction_xs(
            kplus.loc[ids], cc_weekly.loc[ids], kplus_padding, cc_weekly_padding)
        ys = train_set.loc[ids]['income'].to_numpy()
        yield xs, ys


def create_cc_weekly(cc, demo):
    cc_weekly = pd.merge(cc, demo[['id', 'cc_no']], on='cc_no')

    cc_weekly.drop('cc_no', axis=1, inplace=True)

    cc_weekly.sort_values(['id', 'pos_dt'], inplace=True)

    cc_weekly['count'] = 1

    cc_weekly['pos_week_index'] = cc_weekly.apply(
        lambda row: pd.Timestamp(row['pos_dt']).week-1, axis=1)

    cc_weekly = cc_weekly.groupby(['id', 'pos_week_index']).sum().reset_index()
    return cc_weekly
