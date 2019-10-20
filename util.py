import numpy as np

def modified_SMAPE(a, f):
    a , f = np.array(a) ,np.array(f)
    return 100 - (100/len(a)) * np.sum(np.power(np.abs(f - a),2) / np.power(np.minimum(2 * np.abs(a),np.abs(f)) + np.abs(a),2))
