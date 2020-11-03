from keras.callbacks import Callback, CallbackList, BaseLogger, History
import time
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.interpolate import griddata
from keras.models import load_model
from keras.utils import plot_model
import tensorflow as tf
import json
from tqdm import tqdm
from custom_layer import *
from custom_functions import *
import numpy as np
import pandas as pd
import copy
import warnings

np.random.seed(19)
if tf.__version__.startswith('1.'):
    tf.set_random_seed(19)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)
else:
    tf.random.set_seed(19)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

K.set_image_data_format('channels_first')

custom_objects = {'cmse': cmse,
                  'cmc': cmc,
                  'cacc': cacc,
                  'custom_mean_squared_error': custom_mean_squared_error,
                  'custom_mse_cosine': custom_mse_cosine,
                  'custom_binary_accuracy': custom_binary_accuracy,
                  'wasserstein_loss': wasserstein_loss,
                  'wasserstein_accuracy': wasserstein_accuracy,
                  'leaky_relu': leaky_relu,
                  'l2_error_norm': l2_error_norm,
                  'multiply_constant': multiply_constant,
                  'multiply_constant_reciprocal': multiply_constant_reciprocal,
                  'AccumulatorCell': AccumulatorCell}


def register_custom_objects(name, value):
    global custom_objects
    custom_objects[name] = value


def angle_pos(s, offset=0, include_zero=False):
    ap = (np.arange(0, s) + offset) * (360 / s) % 360
    if not include_zero:
        ap = ap[1:]
    return ap


def get_s_d(ap):
    y = -32 * np.sin(ap * (np.pi / 180)) + 32
    x = 32 * np.cos(ap * (np.pi / 180)) + 32
    z = np.ones_like(y)
    s_d = np.array([y, x, z])
    srt = np.lexsort(s_d[-1::-1])
    return s_d, srt


def create_dict(keys, datasets, var_name, func=lambda a: a):
    ret_dict = {}
    for k in keys:
        ret_dict[k] = func(datasets[k][var_name])
    return ret_dict


def dicts_op(keys, dict_list, func=lambda a: a):
    ret_dict = {}
    for k in keys:
        if isinstance(dict_list, list):
            temp = []
            for d in dict_list:
                temp.append(d[k])
            ret_dict[k] = func(temp)
        else:
            ret_dict[k] = func(dict_list[k])
    return ret_dict


MAX_INC = 3


class Dataset:
    def __init__(self, params, name):
        self.name = name
        self.d = params['d'].astype(np.float).reshape((-1, 1))
        self.data_size = len(self.d)
        self.data_label = params.get('data_label', np.ones_like(d)).astype(np.int)
        self.val_size = len(np.where(self.data_label)[0])
        self.train_size = self.data_size - self.val_size
        self.r = params['r'].astype(np.float)
        self.roc = params['roc'].astype(np.float)
        self.theta = params['theta'].astype(np.float)
        self.ca = params['ca'].astype(np.float)
        self.csp = params['csp'].astype(np.float)
        self.count_inc_label = params['count_inc_label'].astype(np.int)
        self.group_inc_label = params['group_inc_label'].astype(np.int)
        self.count_label = self.count_inc_label[self.group_inc_label == 1]
        self.roc_data = np.zeros((self.data_size, MAX_INC))
        self.roc_data[self.count_label != 0, 0] = self.roc[(self.count_inc_label != 0) & (self.group_inc_label == 1)]
        self.roc_data[self.count_label == 2, 1] = self.roc[(self.count_inc_label == 2) & (self.group_inc_label == 2)]
        self.theta_data = np.zeros((self.data_size, MAX_INC))
        self.theta_data[self.count_label != 0, 0] = self.theta[(self.count_inc_label != 0) &
                                                               (self.group_inc_label == 1)]
        self.theta_data[self.count_label == 2, 1] = self.theta[
            (self.count_inc_label == 2) & (self.group_inc_label == 2)]
        self.r_data = np.zeros((self.data_size, MAX_INC))
        self.r_data[self.count_label != 0, 0] = self.r[(self.count_inc_label != 0) & (self.group_inc_label == 1)]
        self.r_data[self.count_label == 2, 1] = self.r[(self.count_inc_label == 2) & (self.group_inc_label == 2)]
        self.ca_data = np.zeros((self.data_size, MAX_INC))
        self.ca_data[self.count_label != 0, 0] = self.ca[(self.count_inc_label != 0) & (self.group_inc_label == 1)]
        self.ca_data[self.count_label == 2, 1] = self.ca[(self.count_inc_label == 2) & (self.group_inc_label == 2)]
        self.csp_data = np.zeros((self.data_size, MAX_INC))
        self.csp_data[self.count_label != 0, 0] = self.csp[(self.count_inc_label != 0) & (self.group_inc_label == 1)]
        self.csp_data[self.count_label == 2, 1] = self.csp[(self.count_inc_label == 2) & (self.group_inc_label == 2)]
        temp = np.flip(np.argsort(self.r_data, axis=1), axis=1)
        temp_func = lambda a: a[a[MAX_INC:].astype(np.int)]
        self.roc_data = np.apply_along_axis(temp_func, 1, np.concatenate([self.roc_data, temp], axis=1))
        self.theta_data = np.apply_along_axis(temp_func, 1, np.concatenate([self.theta_data, temp], axis=1))
        self.r_data = np.apply_along_axis(temp_func, 1, np.concatenate([self.r_data, temp], axis=1))
        self.ca_data = np.apply_along_axis(temp_func, 1, np.concatenate([self.ca_data, temp], axis=1))
        self.csp_data = np.apply_along_axis(temp_func, 1, np.concatenate([self.csp_data, temp], axis=1))
        self.N_count = params.get('N_count').astype(np.int)
        self.freq = params['freq'].astype(np.float).reshape((-1, 1))
        self.num = params['inc_sample_num'].astype(np.int) - 1
        _, self.indices = np.unique(self.num, return_index=True)
        self.inhomo_ct = (self.count_inc_label[self.indices] != 0).astype(np.float)
        self.MUa = params['MUa'].astype(np.float).reshape((-1, 1))
        self.MUsp = params['MUsp'].astype(np.float).reshape((-1, 1))
        self.MU0 = np.concatenate([self.MUa, self.MUsp], axis=1)
        self.inc_areas = None
        self.pos_roc = None
        self.pos_x = None
        self.pos_y = None
        self.pos_data = None
        self.ns = None
        self.MU = None
        self.MU_norm = None
        self.MU_iter_uncorrected = None
        self.MU_iter_corrected = None
        self.MU_deep = None

    def acknowledge_samples(self, samples):
        self.inc_areas = samples['inc_areas'].astype(np.float)
        self.pos_roc = 64 * self.roc_data / self.d
        self.pos_x = self.pos_roc * np.cos(self.theta_data * (np.pi / 180)) + 32
        self.pos_y = -self.pos_roc * np.sin(self.theta_data * (np.pi / 180)) + 32
        temp = np.sqrt(np.diff(self.pos_x) ** 2 + np.diff(self.pos_y) ** 2)
        temp = self.pos_roc[:, 1] > temp[:, 1]
        self.pos_roc[temp, 1] = 0
        self.pos_x[temp, 1] = 0
        self.pos_y[temp, 1] = 0
        self.inc_areas[temp, 0] += self.inc_areas[temp, 1]
        self.inc_areas[temp, 1] = 0
        self.pos_data = np.concatenate([self.pos_x[:, [0]] / 64, self.pos_y[:, [0]] / 64, self.pos_x[:, [1]] / 64,
                                        self.pos_y[:, [1]] / 64], axis=1)
        self.ns = samples['ns_used'].astype(np.int)
        self.MU = np.array([np.concatenate([[x], [y]])
                            for (x, y) in zip(samples['Grid_MUA'], samples['Grid_MUSP'])])
        self.MU_norm = self.MU / np.reshape(self.MU0, self.MU0.shape + (1, 1))

    def acknowledge_iter_results(self, iter_results):
        self.MU_iter_uncorrected = np.array([np.concatenate([[x], [y]])
                                             for (x, y) in zip(iter_results['Grid_MUA_res'],
                                                               iter_results['Grid_MUSP_res'])])
        self.MU_iter_corrected = np.copy(self.MU_iter_uncorrected)
        self.MU_iter_corrected[self.MU_iter_corrected < 0] = 0
        self.MU_iter_corrected[self.MU_iter_corrected > 100] = 100

    def set_MU_deep(self, MU_deep):
        self.MU_deep = MU_deep


temp = np.linspace(-31.5, 31.5, 64) / 32
X, Y = np.meshgrid(temp, temp)
temp = np.linspace(-1, 1, 65)
X2, Y2 = np.meshgrid(temp, temp)
Xf = np.floor(np.abs(X * 32))
Yf = np.floor(np.abs(Y * 32))
boundary_mask = (Xf ** 2 + Yf ** 2 <= 32 ** 2) & ((Xf + 1) ** 2 + (Yf + 1) ** 2 >= 32 ** 2)
Xb = X[boundary_mask]
Yb = Y[boundary_mask]
Theta = np.arctan2(Y, X)
Theta = (Theta * 180 / np.pi) + 180
Thetab = Theta[boundary_mask]
idx = np.argsort(Thetab)
Xb = Xb[idx]
Yb = Yb[idx]
Thetab = Thetab[idx]
Ib = np.arange(len(Xb))
sum_bound = np.sum(boundary_mask)
stats_inc_keys = ('mean', 'max', 'min', 'std', 'median', 'q_p90', 'q_p10', 'tr_range', 'tr_mean')
channel_keys = ('mu_a', 'mu_sp')
final_stats_keys = ('R_cont', 'Ro_size', 'R_size', 'Ro_sep', 'R_sep', 'R_csd', 'ssim')


def get_d_index(ap, indexing=None):
    isnan = np.isnan(ap)
    ap[isnan] = 0
    y = np.sin(ap * (np.pi / 180))
    x = np.cos(ap * (np.pi / 180))
    # return np.array([np.argmin((Xb - xb)**2 + (Yb - yb)**2) for xb, yb in zip(x, y)])
    idx = griddata((Xb, Yb), Ib, (x, y), method='nearest')
    if indexing == 'pythonic':
        idx[isnan] = len(Xb)
    else:
        idx[isnan] = -1
    return idx


def get_s_d_locations(ns, spos=None, dpos=None):
    old_settings = np.seterr(invalid='ignore')
    max_ns_used = max(ns)
    if spos is None:
        spos, dpos = [], []
        for x in ns:
            t1 = np.arange(x)
            t3 = np.kron(t1, np.ones(x - 1))
            # t4 = np.kron(np.ones(x), t1[1:])
            t4 = t3 + np.kron(np.ones(x), t1[1:])
            t3 = t3 * (360 / x) % 360
            t4 = t4 * (360 / x) % 360
            spos.append(t3)
            dpos.append(t4)
    max_len = max_ns_used * (max_ns_used - 1)
    temp3, temp4 = [], []
    for i, (t3, t4) in enumerate(zip(spos, dpos)):
        t3 = t3.astype(np.float)
        t4 = t4.astype(np.float)
        if len(t3) > max_len:
            t3 = t3[:0]
            t4 = t4[:0]
        t2 = np.zeros(max_len - len(t3)) * np.nan
        temp3.append(np.concatenate([t3, t2]))
        temp4.append(np.concatenate([t4, t2]))
    spos = get_d_index(np.array(temp3)) + 1
    dpos = get_d_index(np.array(temp4)) + 1
    spos = np.expand_dims(spos, -1)
    dpos = np.expand_dims(dpos, -1)
    s_d = np.concatenate([spos, dpos], axis=-1).astype(np.int)
    np.seterr(**old_settings)
    return s_d


def get_s_d_locations2(ns):
    old_settings = np.seterr(invalid='ignore')
    max_ns_used = max(ns)
    temp1 = np.array([np.linspace(0, 360, x, endpoint=False) for x in ns])
    temp2 = np.array([np.zeros(max_ns_used * (max_ns_used - 1) - x * (x - 1)) for x in ns]) * np.nan
    temp3 = np.array([np.concatenate([np.kron(t1, np.ones(x - 1)), t2]) for x, t1, t2 in zip(ns, temp1, temp2)])
    temp4 = np.array([np.concatenate([np.kron(np.ones(x), t1[1:]), t2]) for x, t1, t2 in zip(ns, temp1, temp2)])
    temp4 = (temp3 + temp4) % 360
    temp3 = temp3.reshape(temp3.shape + (1,))
    temp4 = temp4.reshape(temp4.shape + (1,))
    s_d = np.concatenate([temp3, temp4], axis=-1)
    y = -32 * np.sin(s_d * (np.pi / 180)) + 32
    x = 32 * np.cos(s_d * (np.pi / 180)) + 32
    temp5 = np.isnan(x[:, :, [0]]) is False
    s_d = np.concatenate([y[:, :, [0]], x[:, :, [0]], y[:, :, [1]], x[:, :, [1]], temp5], axis=-1)
    srt = np.array([np.lexsort(np.transpose(s[:, -1::-1])) for s in s_d])
    s_d[np.isnan(s_d)] = 0
    np.seterr(**old_settings)
    return s_d, srt


print('Loading datasets..')
# train_names = {'training', 'training_16', 'training_16_old'}
test_names = {'uncalibrated', 'calibrated', 'simulation', 'uncalibrated_16', 'calibrated_16', 'simulation_16',
              'uncalibrated_36', 'calibrated_36', 'simulation_36'}
# test_names = {'uncalibrated', 'calibrated', 'simulation', 'uncalibrated_16', 'calibrated_16', 'simulation_16',
#               'uncalibrated2_16', 'calibrated2_16', 'simulation2_16', 'uncalibrated3_16', 'calibrated3_16',
#               'simulation3_16', 'uncalibrated_36', 'calibrated_36', 'simulation_36',
#               'uncalibrated2_36', 'calibrated2_36', 'simulation2_36', 'uncalibrated3_36', 'calibrated3_36',
#               'simulation3_36'}
# samples = {train_names[0]: loadmat('samples_mix.mat'), train_names[1]: loadmat('samples_16.mat'),
#            train_names[2]: loadmat('samples_16_old.mat')}
# param_samples = {train_names[0]: loadmat('param_samples.mat'),
#                  train_names[1]: loadmat('param_samples.mat'),
#                  train_names[2]: loadmat('param_samples_16_old.mat')}
# iter_results = {train_names[0]: loadmat('iterative_results.mat'),
#                 train_names[1]: loadmat('iterative_results.mat'),
#                 train_names[2]: loadmat('iterative_results_16_old.mat')}
exp_data_16 = loadmat('exp_data_rev_16.mat', squeeze_me=True)
exp_data_36 = loadmat('exp_data_rev_36.mat', squeeze_me=True)
exp_data = loadmat('exp_data.mat', squeeze_me=True)


# exp_data_1089 = loadmat('exp_data_1089.mat')
# exp_data_new_clb = loadmat('exp_data_new_clb.mat')


def unwrap_sample(phs, ns):
    for i in range(ns):
        phs[i * (ns - 1):(i + 1) * (ns - 1)] = np.unwrap(phs[i * (ns - 1):(i + 1) * (ns - 1)], axis=0)
    return phs


def add_measurement_dataset(name):
    max_ns = max(ns[name])
    max_data_len = max_ns * (max_ns - 1)
    data = []
    for i, y in enumerate(PHI_meas_lookup[name]):
        a = y.astype(np.complex)
        if len(a) > max_data_len:
            a = a[:0]
        b = np.ones(max_data_len - len(a))
        data.append(np.concatenate([a, b]).reshape((-1, 1)))
    data = np.array(data)
    # data = np.array([np.concatenate([y.astype(np.complex).flatten(),
    #                                  np.ones(max_data_len - y.shape[1])]).reshape((-1, 1))
    #                  for y in PHI_meas_lookup[name].flatten()])
    if name in source_pos:
        s_d = get_s_d_locations(ns[name], source_pos[name], detector_pos[name])
    else:
        s_d = get_s_d_locations(ns[name])
    # s_d = get_s_d_locations(ns[name])
    PHI_meas_raw_seq_pair[name] = data
    log_data = np.log(data)
    phase_data = np.imag(log_data)
    for i, s in enumerate(ns[name]):
        phase_data[i] = unwrap_sample(phase_data[i], s)
    PHI_meas_seq_pair[name] = np.concatenate([np.real(log_data), phase_data, np.array(s_d)], axis=-1)
    if name != 'training':
        data = np.squeeze(data)
        PHI_meas_raw_fixed[name] = data
        log_data = np.squeeze(log_data)
        phase_data = np.squeeze(phase_data)
        PHI_meas_fixed[name] = np.concatenate([np.real(log_data), phase_data], axis=-1)


def add_training_dataset(name, samples_in, param_samples_in, iter_results_in):
    if isinstance(samples_in, str):
        samples_in = loadmat(samples_in, squeeze_me=True)
    if isinstance(param_samples_in, str):
        param_samples_in = loadmat(param_samples_in, squeeze_me=True)
    if isinstance(iter_results_in, str):
        iter_results_in = loadmat(iter_results_in, squeeze_me=True)
    samples[name] = samples_in
    param_samples[name] = param_samples_in
    iter_results[name] = iter_results_in
    datasets[name] = Dataset(param_samples[name], name)
    datasets[name].acknowledge_samples(samples[name])
    datasets[name].acknowledge_iter_results(iter_results[name])
    d[name] = datasets[name].d
    freq[name] = datasets[name].freq
    ns[name] = datasets[name].ns
    roc_data[name] = datasets[name].roc_data
    theta_data[name] = datasets[name].theta_data
    r_data[name] = datasets[name].r_data
    ca_data[name] = datasets[name].ca_data
    csp_data[name] = datasets[name].csp_data
    MUa[name] = datasets[name].MUa
    MUsp[name] = datasets[name].MUsp
    MU0[name] = datasets[name].MU0
    MU[name] = datasets[name].MU
    MU_norm[name] = datasets[name].MU_norm
    MU_norm[name] = datasets[name].MU_norm
    MU_iter_corrected[name] = datasets[name].MU_iter_corrected
    MU_iter_uncorrected[name] = datasets[name].MU_iter_uncorrected
    PHI_meas_lookup[name] = samples[name]['PHI_meas']
    add_measurement_dataset(name)


# print('MAT files loaded.')
train_names, samples, param_samples, iter_results = {}, {}, {}, {}
datasets, d, freq, ns, roc_data, theta_data, r_data, ca_data, csp_data = {}, {}, {}, {}, {}, {}, {}, {}, {}
MUa, MUsp, MU0, MU, MU_norm, MU_iter_corrected, MU_iter_uncorrected = {}, {}, {}, {}, {}, {}, {}
PHI_meas_raw_seq_pair, PHI_meas_seq_pair, PHI_meas_raw_fixed, PHI_meas_fixed = {}, {}, {}, {}
# for k in train_names:
#     datasets[k] = Dataset(param_samples[k], k)
#     datasets[k].acknowledge_samples(samples[k])
#     datasets[k].acknowledge_iter_results(iter_results[k])
#     d[k] = datasets[k].d
#     freq[k] = datasets[k].freq
#     ns[k] = datasets[k].ns
#     roc_data[k] = datasets[k].roc_data
#     MUa[k] = datasets[k].MUa
#     MUsp[k] = datasets[k].MUsp
#     MU0[k] = datasets[k].MU0
#     MU[k] = datasets[k].MU
#     MU_norm[k] = datasets[k].MU_norm
#     MU_norm[k] = datasets[k].MU_norm
#     MU_iter_corrected[k] = datasets[k].MU_iter_corrected
print('Create dictionaries..')
MU_iter_uncorrected = {'uncalibrated': np.array([np.concatenate([[x], [y]])
                                                 for (x, y) in zip(exp_data['Grid_MUA_res'],
                                                                   exp_data['Grid_MUSP_res'])]),
                       'calibrated': np.array([np.concatenate([[x], [y]])
                                               for (x, y) in zip(exp_data['Grid_MUA_res_clb'],
                                                                 exp_data['Grid_MUSP_res_clb'])]),
                       'simulation': np.array([np.concatenate([[x], [y]])
                                               for (x, y) in zip(exp_data['Grid_MUA_res_sim'],
                                                                 exp_data['Grid_MUSP_res_sim'])]),
                       'uncalibrated_16': np.array([np.concatenate([[x], [y]])
                                                    for (x, y) in zip(exp_data_16['Grid_MUA_res'],
                                                                      exp_data_16['Grid_MUSP_res'])]),
                       'calibrated_16': np.array([np.concatenate([[x], [y]])
                                                  for (x, y) in zip(exp_data_16['Grid_MUA_res_clb'],
                                                                    exp_data_16['Grid_MUSP_res_clb'])]),
                       'simulation_16': np.array([np.concatenate([[x], [y]])
                                                  for (x, y) in zip(exp_data_16['Grid_MUA_res_sim'],
                                                                    exp_data_16['Grid_MUSP_res_sim'])]),
                       'uncalibrated_36': np.array([np.concatenate([[x], [y]])
                                                    for (x, y) in zip(exp_data_36['Grid_MUA_res'],
                                                                      exp_data_36['Grid_MUSP_res'])]),
                       'calibrated_36': np.array([np.concatenate([[x], [y]])
                                                  for (x, y) in zip(exp_data_36['Grid_MUA_res_clb'],
                                                                    exp_data_36['Grid_MUSP_res_clb'])]),
                       'simulation_36': np.array([np.concatenate([[x], [y]])
                                                  for (x, y) in zip(exp_data_36['Grid_MUA_res_sim'],
                                                                    exp_data_36['Grid_MUSP_res_sim'])])}
MU_iter_uncorrected['uncalibrated2_16'] = MU_iter_uncorrected['uncalibrated_16']
MU_iter_uncorrected['calibrated2_16'] = MU_iter_uncorrected['calibrated_16']
MU_iter_uncorrected['simulation2_16'] = MU_iter_uncorrected['simulation_16']
MU_iter_uncorrected['uncalibrated3_16'] = MU_iter_uncorrected['uncalibrated_16']
MU_iter_uncorrected['calibrated3_16'] = MU_iter_uncorrected['calibrated_16']
MU_iter_uncorrected['simulation3_16'] = MU_iter_uncorrected['simulation_16']
MU_iter_uncorrected['uncalibrated2_36'] = MU_iter_uncorrected['uncalibrated_36']
MU_iter_uncorrected['calibrated2_36'] = MU_iter_uncorrected['calibrated_36']
MU_iter_uncorrected['simulation2_36'] = MU_iter_uncorrected['simulation_36']
MU_iter_uncorrected['uncalibrated3_36'] = MU_iter_uncorrected['uncalibrated_36']
MU_iter_uncorrected['calibrated3_36'] = MU_iter_uncorrected['calibrated_36']
MU_iter_uncorrected['simulation3_36'] = MU_iter_uncorrected['simulation_36']

PHI_meas_lookup = {'uncalibrated': exp_data['PHI_meas'], 'calibrated': exp_data['PHI_CLB_meas'],
                   'simulation': exp_data['PHI_meas_sim'], 'uncalibrated_16': exp_data_16['PHI_meas'],
                   'calibrated_16': exp_data_16['PHI_CLB_meas'], 'simulation_16': exp_data_16['PHI_meas_sim'],
                   'uncalibrated_36': exp_data_36['PHI_meas'], 'calibrated_36': exp_data_36['PHI_CLB_meas'],
                   'simulation_36': exp_data_36['PHI_meas_sim']}
source_pos = {'uncalibrated_16': exp_data_16['source_pos'], 'calibrated_16': exp_data_16['source_pos'],
              'simulation_16': exp_data_16['source_pos_sim'], 'uncalibrated_36': exp_data_36['source_pos'],
              'calibrated_36': exp_data_36['source_pos'], 'simulation_36': exp_data_36['source_pos_sim']}
detector_pos = {'uncalibrated_16': exp_data_16['detector_pos'], 'calibrated_16': exp_data_16['detector_pos'],
                'simulation_16': exp_data_16['detector_pos_sim'], 'uncalibrated_36': exp_data_36['detector_pos'],
                'calibrated_36': exp_data_36['detector_pos'], 'simulation_36': exp_data_36['detector_pos_sim']}
# PHI_meas_lookup = {'uncalibrated': exp_data['PHI_meas'], 'calibrated': exp_data['PHI_CLB_meas'],
#                    'simulation': exp_data['PHI_meas_sim'], 'uncalibrated_16': exp_data_16['PHI_meas'],
#                    'calibrated_16': exp_data_16['PHI_CLB_meas'], 'simulation_16': exp_data_16['PHI_meas_sim'],
#                    'uncalibrated2_16': exp_data_16['PHI_meas2'], 'calibrated2_16': exp_data_16['PHI_CLB_meas2'],
#                    'simulation2_16': exp_data_16['PHI_meas_sim'], 'uncalibrated3_16': exp_data_16['PHI_meas3'],
#                    'calibrated3_16': exp_data_16['PHI_CLB_meas3'], 'simulation3_16': exp_data_16['PHI_meas_sim'],
#                    'uncalibrated_36': exp_data_36['PHI_meas'], 'calibrated_36': exp_data_36['PHI_CLB_meas'],
#                    'simulation_36': exp_data_36['PHI_meas_sim'], 'uncalibrated2_36': exp_data_36['PHI_meas2'],
#                    'calibrated2_36': exp_data_36['PHI_CLB_meas2'], 'simulation2_36': exp_data_36['PHI_meas_sim'],
#                    'uncalibrated3_36': exp_data_36['PHI_meas3'], 'calibrated3_36': exp_data_36['PHI_CLB_meas3'],
#                    'simulation3_36': exp_data_36['PHI_meas_sim']}
# source_pos = {'uncalibrated_16': exp_data_16['source_pos'], 'calibrated_16': exp_data_16['source_pos'],
#               'simulation_16': exp_data_16['source_pos_sim'], 'uncalibrated2_16': exp_data_16['source_pos_sim'],
#               'calibrated2_16': exp_data_16['source_pos_sim'], 'simulation2_16': exp_data_16['source_pos_sim'],
#               'uncalibrated3_16': exp_data_16['source_pos'], 'calibrated3_16': exp_data_16['source_pos'],
#               'simulation3_16': exp_data_16['source_pos_sim'], 'uncalibrated_36': exp_data_36['source_pos'],
#               'calibrated_36': exp_data_36['source_pos'], 'simulation_36': exp_data_36['source_pos_sim'],
#               'uncalibrated2_36': exp_data_36['source_pos_sim'], 'calibrated2_36': exp_data_36['source_pos_sim'],
#               'simulation2_36': exp_data_36['source_pos_sim'], 'uncalibrated3_36': exp_data_36['source_pos'],
#               'calibrated3_36': exp_data_36['source_pos'], 'simulation3_36': exp_data_36['source_pos_sim']}
# detector_pos = {'uncalibrated_16': exp_data_16['detector_pos'], 'calibrated_16': exp_data_16['detector_pos'],
#                 'simulation_16': exp_data_16['detector_pos_sim'], 'uncalibrated2_16': exp_data_16['detector_pos_sim'],
#                 'calibrated2_16': exp_data_16['detector_pos_sim'], 'simulation2_16': exp_data_16['detector_pos_sim'],
#                 'uncalibrated3_16': exp_data_16['detector_pos'], 'calibrated3_16': exp_data_16['detector_pos'],
#                 'simulation3_16': exp_data_16['detector_pos_sim'], 'uncalibrated_36': exp_data_36['detector_pos'],
#                 'calibrated_36': exp_data_36['detector_pos'], 'simulation_36': exp_data_36['detector_pos_sim'],
#                 'uncalibrated2_36': exp_data_36['detector_pos_sim'], 'calibrated2_36': exp_data_36['detector_pos_sim'],
#                 'simulation2_36': exp_data_36['detector_pos_sim'], 'uncalibrated3_36': exp_data_36['detector_pos'],
#                 'calibrated3_36': exp_data_36['detector_pos'], 'simulation3_36': exp_data_36['detector_pos_sim']}
# PHI_meas_lookup_id = {'training': 0, 'training_16': 0, 'training_16_old': 0, 'uncalibrated': 0, 'calibrated': 0,
#                       'simulation': 0, 'uncalibrated_1089': 0, 'calibrated_1089': 0, 'simulation_1089': 0,
#                       'uncalibrated_new_clb': 0, 'calibrated_new_clb': 0, 'simulation_new_clb': 0}

add_training_dataset('training', 'samples.mat', 'param_samples.mat', 'iterative_results.mat')
add_training_dataset('training_16', 'samples_16.mat', param_samples['training'], 'iterative_results_16_rev.mat')
add_training_dataset('training_36', 'samples_36.mat', param_samples['training'], 'iterative_results_36.mat')
add_training_dataset('training_16_old', 'samples_16_old.mat', 'param_samples_16_old.mat',
                     'iterative_results_16_old.mat')
temp_lookup = {'exp': exp_data, 'exp_16': exp_data_16, 'exp_36': exp_data_36}
temp_d, temp_freq, temp_ns = {}, {}, {}
temp_roc_data, temp_theta_data, temp_r_data, temp_ca_data, temp_csp_data = {}, {}, {}, {}, {}
temp_MUa, temp_MUsp, temp_MU = {}, {}, {}
for k, v in temp_lookup.items():
    temp_len = len(v['MUa'])
    temp_d[k] = v['d'].astype(np.float).reshape((-1, 1))
    if k == 'exp':
        temp_freq[k] = np.ones((temp_len, 1)) * 20e6
        temp_ns[k] = np.ones(temp_len, dtype=np.int) * 16
    else:
        temp_freq[k] = v['freq'].astype(np.float).reshape((-1, 1))
        temp_ns[k] = np.ones(temp_len, dtype=np.int) * int(k[-2:])
    temp_roc_data[k] = np.zeros((temp_len, MAX_INC))
    temp_theta_data[k] = np.zeros((temp_len, MAX_INC))
    temp_r_data[k] = np.zeros((temp_len, MAX_INC))
    temp_ca_data[k] = np.zeros((temp_len, MAX_INC))
    temp_csp_data[k] = np.zeros((temp_len, MAX_INC))
    for i, (t_roc, t_theta, t_r, t_ca, t_csp) in enumerate(zip(v['roc'], v['theta'], v['r'], v['ca'], v['csp'])):
        try:
            l = len(t_roc)
        except TypeError:
            l = 1
        t_roc = np.array(t_roc, dtype=np.float).flatten()
        t_theta = np.array(t_theta, dtype=np.float).flatten()
        t_r = np.array(t_r, dtype=np.float).flatten()
        t_ca = np.array(t_ca, dtype=np.float).flatten()
        t_csp = np.array(t_csp, dtype=np.float).flatten()
        temp_roc_data[k][i, :l] = t_roc
        temp_theta_data[k][i, :l] = t_theta
        temp_r_data[k][i, :l] = t_r
        temp_ca_data[k][i, :l] = t_ca
        temp_csp_data[k][i, :l] = t_csp
        temp = np.flip(np.argsort(temp_r_data[k], axis=1), axis=1)
        temp_func = lambda a: a[a[MAX_INC:].astype(np.int)]
        temp_roc_data[k] = np.apply_along_axis(temp_func, 1, np.concatenate([temp_roc_data[k], temp], axis=1))
        temp_theta_data[k] = np.apply_along_axis(temp_func, 1, np.concatenate([temp_theta_data[k], temp], axis=1))
        temp_r_data[k] = np.apply_along_axis(temp_func, 1, np.concatenate([temp_r_data[k], temp], axis=1))
        temp_ca_data[k] = np.apply_along_axis(temp_func, 1, np.concatenate([temp_ca_data[k], temp], axis=1))
        temp_csp_data[k] = np.apply_along_axis(temp_func, 1, np.concatenate([temp_csp_data[k], temp], axis=1))
    temp_MUa[k] = v['MUa'].astype(np.float).reshape((-1, 1))
    temp_MUsp[k] = v['MUsp'].astype(np.float).reshape((-1, 1))
    temp_MU[k] = np.array([np.concatenate([[x], [y]]) for (x, y) in zip(v['Grid_MUA'], v['Grid_MUSP'])])
for k in test_names:
    if k.endswith('_16'):
        k2 = 'exp_16'
    elif k.endswith('_36'):
        k2 = 'exp_36'
    else:
        k2 = 'exp'
    d[k] = temp_d[k2]
    freq[k] = temp_freq[k2]
    ns[k] = temp_ns[k2]
    roc_data[k] = temp_roc_data[k2]
    theta_data[k] = temp_theta_data[k2]
    r_data[k] = temp_r_data[k2]
    ca_data[k] = temp_ca_data[k2]
    csp_data[k] = temp_csp_data[k2]
    MUa[k] = temp_MUa[k2]
    MUsp[k] = temp_MUsp[k2]
    MU0[k] = np.concatenate([MUa[k], MUsp[k]], axis=1)
    MU[k] = temp_MU[k2]
    MU_norm[k] = MU[k] / np.reshape(MU0[k], MU0[k].shape + (1, 1))
    MU_iter_corrected[k] = MU_iter_uncorrected[k].copy()
    MU_iter_corrected[k][MU_iter_corrected[k] < 0] = 0
    MU_iter_corrected[k][MU_iter_corrected[k] > 100] = 100
    add_measurement_dataset(k)
MU_iter = MU_iter_corrected
MU0_deep = MU0.copy()
# X = samples[training_names[0]]['X']
# Y = samples[training_names[0]]['Y']
mask = loadmat('mask.mat')['mask'].astype(np.bool)
empty_image = mask / mask * ~mask
c1_kmeans = 1
c2_kmeans = 100
X_kmeans = (X[mask].reshape(-1, 1) + 1) * c1_kmeans
Y_kmeans = (Y[mask].reshape(-1, 1) + 1) * c1_kmeans
boundary_kmeans = (mask & boundary_mask)[mask]
# del temp_d, temp_freq, temp_ns, temp_MUa, temp_MUsp, temp_MU
PHI_0_flag, n_flag, PHI_meas_seq_pair_masked = {}, {}, {}
for k, v in PHI_meas_seq_pair.items():
    s = v.shape
    PHI_0_flag[k] = np.zeros(s[:2], dtype=np.bool)
    s2 = ns[k] * (ns[k] - 1)
    n_flag[k] = (s2 * np.random.random(size=ns[k].shape) * 0.2).astype(np.int)
    for i in range(s[0]):
        PHI_0_flag[k][i, np.random.randint(s2[i], size=n_flag[k][i])] = 1
    PHI_meas_seq_pair_masked[k] = np.copy(v)
    PHI_meas_seq_pair_masked[k][PHI_0_flag[k], :] = 0
print('Dictionaries created.')

# try:
#     PHI_meas_raw0 = loadmat('PHI_meas_raw0.mat')
#     PHI_meas0 = loadmat('PHI_meas0.mat')
# except FileNotFoundError:
# PHI_meas_raw0, PHI_meas0 = {}, {}
# for k in ns.keys():
#     max_ns = max(ns[k])
#     data = []
#     s_d = []
#     for s, y in zip(ns[k], PHI_meas_lookup[k]):
#         temp = y[0].astype(np.complex).flatten()
#         ap = angle_pos(s, include_zero=True)
#         d_index = get_d_index(ap) + 1
#         # dat = np.ones((max_ns, sum_bound), dtype=np.complex) * 1e-20
#         # dat[len(ap):] = 1
#         dat = np.ones((max_ns, sum_bound), dtype=np.complex)
#         for i in range(len(ap)):
#             di = get_d_index(angle_pos(s, offset=i))
#             dat[i, di] = temp[i * (s - 1):(i + 1) * (s - 1)]
#         data.append(dat)
#         s_d.append(np.concatenate([d_index.reshape((-1, 1)), np.zeros((max_ns - s, 1))]))
#     data = np.array(data)
#     PHI_meas_raw0[k] = data
#     data = np.log(data)
#     PHI_meas0[k] = np.concatenate([np.real(data), np.imag(data), np.array(s_d)], axis=-1)
# print('Prepare measurement data..')
# savemat('PHI_meas_raw0.mat', PHI_meas_raw0)
# savemat('PHI_meas0.mat', PHI_meas0)

# try:
#     PHI_meas_raw1 = loadmat('PHI_meas_raw1.mat')
#     PHI_meas1 = loadmat('PHI_meas1.mat')
# except FileNotFoundError:
# PHI_meas_raw1, PHI_meas1 = {}, {}
# ns_target = 36
# target_angle_pos = angle_pos(ns_target)
# for k in ns.keys():
#     max_ns = max(ns[k])
#     data = []
#     s_d = []
#     for s, y in zip(ns[k], PHI_meas_lookup[k]):
#         temp = y[0].astype(np.complex).flatten()
#         ap = angle_pos(s, include_zero=True)
#         data.append(
#             np.concatenate([np.array([np.interp(target_angle_pos, angle_pos(s, offset=i),
#                                                 temp[i * (s - 1):(i + 1) * (s - 1)])
#                                       for i in range(len(ap))]), np.ones((max_ns - len(ap), ns_target - 1))]))
#         tmp_s_d, srt = get_s_d(ap)
#         s_d.append(np.concatenate([np.transpose(tmp_s_d), np.zeros((max_ns - s, 3))]))
#         srt = np.concatenate([srt, np.arange(s, max_ns)])
#         data[-1] = data[-1][srt]
#         s_d[-1] = s_d[-1][srt]
#     data = np.array(data)
#     PHI_meas_raw1[k] = data
#     data = np.log(data)
#     PHI_meas1[k] = np.concatenate([np.real(data), np.imag(data), np.array(s_d)], axis=-1)
print('Prepare measurement data..')
# savemat('PHI_meas_raw1.mat', PHI_meas_raw1)
# savemat('PHI_meas1.mat', PHI_meas1)

# try:
#     PHI_meas_raw2 = loadmat('PHI_meas_raw2.mat')
#     PHI_meas2 = loadmat('PHI_meas2.mat')
# except FileNotFoundError:
# PHI_meas_raw2, PHI_meas2 = {}, {}
# for k in ns.keys():
#     max_ns = max(ns[k])
#     max_data_len = max_ns * (max_ns - 1)
#     data = []
#     # data = np.array([np.concatenate([y[0].astype(np.complex).flatten(),
#     #                                  np.ones(max_data_len - y[0].shape[1])]).reshape((-1, 1))
#     #                  for y in PHI_meas_lookup[k]])
#     s_d, srt = get_s_d_locations2(ns[k])
#     # s_d = get_s_d_locations(ns[k])
#     for i, (y, srt_) in enumerate(zip(PHI_meas_lookup[k], srt)):
#         temp = y[0].astype(np.complex).flatten()
#         data.append(np.concatenate([temp, np.ones(max_data_len - y[0].shape[1])])
#                     .reshape((-1, 1)))
#         data[-1] = data[-1][srt_]
#         s_d[i] = s_d[i][srt_]
#     data = np.array(data)
#     PHI_meas_raw2[k] = data
#     data = np.log(data)
#     PHI_meas2[k] = np.concatenate([np.real(data), np.unwrap(np.imag(data), axis=1), np.array(s_d)], axis=-1)
# PHI_meas_raw2, PHI_meas2 = {}, {}
# for k in ns.keys():
#     max_ns = max(ns[k])
#     max_data_len = max_ns * (max_ns - 1)
#     data = []
#     s_d = []
#     for s, y in zip(ns[k], PHI_meas_lookup[k]):
#         temp = y[0].astype(np.complex).flatten()
#         data.append(np.concatenate([temp, np.ones(max_data_len - y[0].shape[1])]).reshape((-1, 1)))
#         tmp_s_d = np.zeros((max_data_len, 2), dtype=np.int)
#         d_index = get_d_index(angle_pos(s, include_zero=True)) + 1
#         for i in range(len(d_index)):
#             di = get_d_index(angle_pos(s, offset=i)) + 1
#             tmp_s_d[i * (s - 1):(i + 1) * (s - 1), 0] = d_index[i]
#             tmp_s_d[i * (s - 1):(i + 1) * (s - 1), 1] = di
#         s_d.append(tmp_s_d)
#     data = np.array(data)
#     PHI_meas_raw2[k] = data
#     data = np.log(data)
#     PHI_meas2[k] = np.concatenate([np.real(data), np.imag(data), np.array(s_d)], axis=-1)
# savemat('PHI_meas_raw2.mat', PHI_meas_raw2)
# savemat('PHI_meas2.mat', PHI_meas2)

# try:
#     PHI_meas_raw_16 = loadmat('PHI_meas_raw_16.mat')
#     PHI_meas_16 = loadmat('PHI_meas_16.mat')
# except FileNotFoundError:
# PHI_meas_raw_16, PHI_meas_16 = {}, {}
# for k in ns.keys():
#     if k == 'training':
#         continue
#     data = np.array([y[0].astype(np.complex).flatten() for y in PHI_meas_lookup[k]])
#     PHI_meas_raw_16[k] = data
#     data = np.log(data)
#     PHI_meas_16[k] = np.concatenate([np.real(data), np.unwrap(np.imag(data))], axis=-1)
#     # savemat('PHI_meas_raw_16.mat', PHI_meas_raw_16)
#     # savemat('PHI_meas_16.mat', PHI_meas_16)

model = None
input_for_model = None
hist_data = None
PREPROCESSING_MODE = None
PHI_meas_raw = None
PHI_meas = None
seq_len = None
p_theta = np.linspace(0, 2 * np.pi, 181)
MU_deep = {}
key = 'training_16'
noise_max = np.max(param_samples[key]['t_noise']) / 100
noise_add_max = 10 ** (np.max(-param_samples[key]['t_noise_db']) / 20)
const_freq = np.mean(param_samples[key]['t_freq'])
const_d = np.mean(param_samples[key]['t_d'])
print('All datasets loaded.')


# PHI_meas = {'training': np.array([np.concatenate([np.concatenate([
#     np.real(np.log(x[0].astype(np.complex))).reshape(-1, 1),
#     np.imag(np.log(x[0].astype(np.complex))).reshape(-1, 1)], axis=1),
#     np.zeros((max_data_len-x[0].shape[1], 2))]) for x in samples['PHI_meas']]),
#             'uncalibrated': np.array([np.concatenate([np.real(np.log(x[0])).reshape(-1, 1),
#                                                       np.imag(np.log(x[0])).reshape(-1, 1)], axis=1)
#                                       for x in exp_data]),
#             'calibrated': np.array([np.concatenate([np.real(np.log(x[10])).reshape(-1, 1),
#                                                     np.imag(np.log(x[10])).reshape(-1, 1)], axis=1) for x in exp_data]),
#             'simulation': np.array([np.concatenate([np.real(np.log(x[16])).reshape(-1, 1),
#                                                     np.imag(np.log(x[16])).reshape(-1, 1)], axis=1) for x in exp_data])}
# PHI_meas = {}
# for k, v in PHI_meas_raw.items():
#     temp = np.concatenate([np.log(v).view('float64'), s_d[k]], axis=-1)
#     PHI_meas[k] = np.array([x[i, :] for x, i in zip(temp, srt[k])])
# PHI_meas = np.array([np.concatenate([[np.real(x[0].flatten())], [np.imag(x[0].flatten())]]) for x in PHI_meas])
# MU['uncalibrated'] = np.array([np.concatenate(([x[0]], [y[0]]))
#                                for (x, y) in zip(exp_data['Grid_MUA'], exp_data['Grid_MUSP'])])
# MU_iter_uncorrected['uncalibrated'] = np.array([np.concatenate(([x[0]], [y[0]]))
#                                                 for (x, y) in zip(exp_data['Grid_MUA_res'],
#                                                                   exp_data['Grid_MUSP_res'])])
# PHI_meas['uncalibrated'] = np.array([np.concatenate([[np.real(np.log(x[0]).flatten())],
#                                                      [np.imag(np.log(x[0]).flatten())]])
#                                      for x in exp_data['PHI_meas']])
# MU_iter_uncorrected['calibrated'] = np.array([np.concatenate(([x[0]], [y[0]]))
#                                               for (x, y) in zip(exp_data['Grid_MUA_res_clb'],
#                                                                 exp_data['Grid_MUSP_res_clb'])])
# PHI_meas['calibrated'] = np.array([np.concatenate([np.real(np.log(x[0]).flatten()),
#                                                    np.imag(np.log(x[0]).flatten())])
#                                    for x in exp_data['PHI_CLB_meas']])
# MU_iter_uncorrected['simulation'] = np.array([np.concatenate(([x[0]], [y[0]]))
#                                               for (x, y) in zip(exp_data['Grid_MUA_res_sim'],
#                                                                 exp_data['Grid_MUSP_res_sim'])])
# PHI_meas['simulation'] = np.array([np.concatenate((np.real(np.log(x[0]).flatten()), np.imag(np.log(x[0]).flatten())))
#                                    for x in exp_data['PHI_meas_sim']])


# def normalize_phi(PHI_meas):
#     l = int(PHI_meas.shape[1] / 2)
#     idx1 = np.arange(7, l, 15)
#     idx2 = (np.arange(16).reshape((16, 1)) * 15 + np.array([0, 1, 2, 3, 11, 12, 13, 14])).flatten()
#     x1 = PHI_meas[:, :l]
#     x2 = PHI_meas[:, l:]
#     beta1 = np.max(x1, axis=1, keepdims=True)
#     opt_sel = np.max(x1[:, idx1], axis=1, keepdims=True)
#     gamma1 = beta1 - opt_sel
#     beta2 = np.min(x2[:, idx2], axis=1, keepdims=True)
#     opt_sel = np.max(x2[:, idx2], axis=1, keepdims=True)
#     gamma2 = (opt_sel - beta2) / 0.63
#     xn1 = (x1 - beta1) / gamma1
#     xn2 = (x2 - beta2) / gamma2
#     params = np.concatenate([gamma1, beta1, gamma2, beta2], axis=1)
#     return np.concatenate([xn1, xn2], axis=1), params
#
#
# PHI_meas_norm = {}
# PHI_meas_params = {}
# for k, v in PHI_meas.items():
#     PHI_meas_norm[k], PHI_meas_params[k] = normalize_phi(v)


def set_preprocessing_mode(mode):
    global PREPROCESSING_MODE, PHI_meas_raw, PHI_meas, seq_len
    if mode == 'FIXED':
        PHI_meas_raw = PHI_meas_raw_fixed
        PHI_meas = PHI_meas_fixed
        seq_len = {}
        for k, v in ns.items():
            seq_len[k] = v * (v - 1) * 2
    # elif mode == 'SEQUENTIAL_SOURCE_FULL_BOUNDARY':
    #     PHI_meas_raw = PHI_meas_raw0
    #     PHI_meas = PHI_meas0
    #     seq_len = ns
    # elif mode == 'SEQUENTIAL_SOURCE':
    #     PHI_meas_raw = PHI_meas_raw1
    #     PHI_meas = PHI_meas1
    #     seq_len = ns
    elif mode == 'SEQUENTIAL':
        PHI_meas_raw = PHI_meas_raw_seq_pair
        PHI_meas = PHI_meas_seq_pair
        seq_len = {}
        for k, v in ns.items():
            seq_len[k] = v * (v - 1)
    else:
        raise Exception('Preprocessing mode not allowed.')
    PREPROCESSING_MODE = mode


# set_preprocessing_mode('SEQUENTIAL')


set_preprocessing_mode('FIXED')


# set_preprocessing_mode('SEQUENTIAL_SOURCE_FULL_BOUNDARY')


def get_raw_measurements(meas):
    if PREPROCESSING_MODE == 'FIXED':
        real = meas[:, :(meas.shape[1] // 2)]
        imag = meas[:, (meas.shape[1] // 2):]
    # elif PREPROCESSING_MODE == 'SEQUENTIAL_SOURCE':
    #     real = meas[:, :, :(ns_target - 1)]
    #     imag = meas[:, :, (ns_target - 1):((ns_target - 1) * 2)]
    elif PREPROCESSING_MODE == 'SEQUENTIAL':
        real = meas[:, :, 0:1]
        imag = meas[:, :, 1:2]
    else:
        raise Exception('Preprocessing mode not allowed.')
    return np.exp(real + imag * 1j)


def set_raw_measurements(meas, meas_raw):
    log_meas_raw = np.log(meas_raw)
    if PREPROCESSING_MODE == 'FIXED':
        meas[:, :(meas.shape[1] // 2)] = np.real(log_meas_raw)
        meas[:, (meas.shape[1] // 2):] = np.imag(log_meas_raw)
    # elif PREPROCESSING_MODE == 'SEQUENTIAL_SOURCE':
    #     mask = meas[:, :, -1]
    #     meas[:, :, :(ns_target - 1)] = meas[:, :, :(ns_target - 1)] * (1 - mask) + np.real(log_meas_raw) * mask
    #     meas[:, :, (ns_target - 1):((ns_target - 1) * 2)] =\
    #         meas[:, :, (ns_target - 1):((ns_target - 1) * 2)] * (1 - mask) + np.imag(log_meas_raw) * mask
    elif PREPROCESSING_MODE == 'SEQUENTIAL':
        mask = np.all(meas[:, :, 2:], axis=-1, keepdims=True)
        meas[:, :, 0:1] = meas[:, :, 0:1] * (1 - mask) + np.real(log_meas_raw) * mask
        meas[:, :, 1:2] = meas[:, :, 1:2] * (1 - mask) + np.imag(log_meas_raw) * mask
    else:
        raise Exception('Preprocessing mode not allowed.')
    return meas


def correct_MU_iter(b):
    global MU_iter
    if b:
        MU_iter = MU_iter_corrected
    else:
        MU_iter = MU_iter_uncorrected


def mask_image(image, mask_value):
    image = image.copy()
    image[~mask] = mask_value
    return image


def obtain_results(param_model=model, param_input=input_for_model, weights=None, sel=slice(2), sel_bg=slice(4, 6),
                   print_summary=False, exclude_keys=[]):
    global MU_deep, deep_output, input_for_model
    set_model(param_model)
    if weights is not None:
        set_weights(weights)
    if param_input is not input_for_model:
        if isinstance(param_input, list):
            input_for_model = {}
            for k in param_input[0].keys():
                if k in exclude_keys:
                    continue
                input_for_model[k] = [inp[k] for inp in param_input]
        else:
            input_for_model = param_input.copy()
            for k in exclude_keys:
                input_for_model.pop(k)
    deep_output = {}
    for k, v in input_for_model.items():
        deep_output[k] = model.predict(v)
    if print_summary:
        model.summary()
    if isinstance(deep_output[list(deep_output.keys())[0]], list):
        for k, v in deep_output.items():
            if isinstance(sel, int):
                MU_deep[k] = v[sel]
            elif isinstance(sel, slice):
                MU_deep[k] = np.concatenate(v[sel], axis=1)
            else:
                MU_deep[k] = np.concatenate([v[i] for i in sel], axis=1)
            if isinstance(sel_bg, int):
                MU0_deep[k] = v[sel_bg]
            elif isinstance(sel_bg, slice):
                MU0_deep[k] = np.concatenate(v[sel_bg], axis=1)
            else:
                MU0_deep[k] = np.concatenate([v[i] for i in sel_bg], axis=1)
    else:
        MU_deep = deep_output
    for k in deep_output.keys():
        if k in train_names:
            datasets[k].set_MU_deep(MU_deep[k])
    return MU_deep


def get_current_results():
    return MU_deep


def get_model():
    return model


def set_model(param_model):
    global model
    if param_model is not model:
        if isinstance(param_model, str):
            model = load_model(param_model, custom_objects=custom_objects)
        elif param_model is not None:
            model = param_model


def set_weights(weights):
    global model
    if isinstance(weights, str):
        model.load_weights(weights)
    else:
        model.set_weights(weights)


def set_hist_data(hist_data_input):
    global hist_data
    if hist_data_input is not hist_data:
        if isinstance(hist_data_input, str):
            with open(hist_data_input, 'r') as f:
                hist_data = json.load(f)
        else:
            hist_data = hist_data


def plot_loss(save_name=None, hist_data_input=hist_data, loss_list=['loss', 'val_loss']):
    set_hist_data(hist_data_input)
    for l in loss_list:
        plt.plot(hist_data['history'][l])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loss_list, loc='upper left')
    plt.show()
    if save_name is not None:
        plt.savefig(save_name)


def model_visualization(filename, model_input=model):
    if model_input is not model:
        if isinstance(model_input, str):
            viz_model = load_model(model_input, custom_objects=custom_objects)
        else:
            viz_model = model_input
    plot_model(viz_model, to_file=filename, show_shapes=True)


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []
        self.train_time_start = time.time()

    def on_train_end(self, logs={}):
        self.train_time_end = time.time()
        self.train_time = self.train_time_end - self.train_time_start

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.perf_counter()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.perf_counter() - self.epoch_time_start)


def ground_truth_vs_estimated(data, data_est, centers, plot=False):
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    centers = np.concatenate([np.array([[0, 0, 1, 1]]), centers], axis=0)
    centers_ = np.zeros_like(centers)
    centers_[:, 0] = centers[:, 0] * np.cos(centers[:, 1] * (np.pi / 180))
    centers_[:, 1] = centers[:, 0] * np.sin(centers[:, 1] * (np.pi / 180))
    centers_[:, :2] = (centers_[:, :2] + 1) * c1_kmeans
    data_ = data.transpose((1, 2, 0))[mask]
    l = data_.shape[-1]
    data_ = data_.reshape(-1, l)
    temp_min = np.min(data_, axis=0)
    data_kmeans = data_ - temp_min
    centers_[:, 2:] *= temp_min
    centers_[:, 2:] -= temp_min
    temp = np.max(data_kmeans, axis=0)
    temp[temp == 0] = 1
    data_kmeans *= c2_kmeans / temp
    centers_[:, 2:] *= c2_kmeans / temp
    data_est_ = data_est.transpose((1, 2, 0))[mask]
    data_est_ = data_est_.reshape(-1, data_est_.shape[-1])
    n_clusters = len(centers_)
    kmeans = KMeans(n_clusters=n_clusters, init=centers_).fit(np.concatenate([X_kmeans, Y_kmeans, data_kmeans],
                                                                             axis=-1))
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    clusters = np.unique(labels)
    temp = np.argmax(np.bincount(labels[boundary_kmeans]))
    if temp != 0:
        temp_flag = labels == temp
        labels[labels < temp] += 1
        labels[temp_flag] = 0
        centers = np.concatenate([centers[temp:temp + 1], centers[:temp], centers[temp + 1:]])
    # filt = np.where(np.abs(kmeans.cluster_centers_) < 1e-4)[0]
    # clusters = np.setdiff1d(clusters, filt)
    # xM = []
    # for i in clusters:
    #     xM.append(np.max(data_[labels == i]))
    # xM = np.array(xM)
    # x = np.array([np.mean(data_[labels == i]) for i in clusters])
    # x_est = np.array([np.mean(data_est_[labels == i]) for i in clusters])
    # xM = np.array([np.max(data_[labels == i]) for i in clusters])
    # xM_est = np.array([np.max(data_est_[labels == i]) for i in clusters])
    # xm = np.array([np.min(data_[labels == i]) for i in clusters])
    # xm_est = np.array([np.min(data_est_[labels == i]) for i in clusters])
    # xs = np.array([np.std(data_[labels == i]) for i in clusters])
    # xs_est = np.array([np.std(data_est_[labels == i]) for i in clusters])
    # s = np.array([(labels == i).size for i in clusters])
    stats_inc_orig = pd.DataFrame(index=clusters, columns=pd.MultiIndex.from_product([stats_inc_keys, channel_keys]))
    stats_inc_est = stats_inc_orig.copy()
    # di = data_[(labels != 0) | (len(clusters) == 1)]
    # di_est = data_est_[(labels != 0) | (len(clusters) == 1)]
    for i in clusters:
        d = data_[labels == i]
        d_est = data_est_[labels == i]
        if i == 0:
            d0 = d
            d0_est = d_est
        s = np.where(labels == i)[0].size
        stats_inc_orig.loc[i, 'mean'] = np.mean(d, axis=0)
        stats_inc_est.loc[i, 'mean'] = np.mean(d_est, axis=0)
        stats_inc_orig.loc[i, 'max'] = np.max(d, axis=0)
        stats_inc_est.loc[i, 'max'] = np.max(d_est, axis=0)
        stats_inc_orig.loc[i, 'min'] = np.min(d, axis=0)
        stats_inc_est.loc[i, 'min'] = np.min(d_est, axis=0)
        stats_inc_orig.loc[i, 'std'] = np.std(d, axis=0)
        stats_inc_est.loc[i, 'std'] = np.std(d_est, axis=0)
        stats_inc_orig.loc[i, 'median'] = np.median(d, axis=0)
        stats_inc_est.loc[i, 'median'] = np.median(d_est, axis=0)
        q_p90 = np.quantile(d, 0.9, interpolation='nearest', axis=0)
        q_p90_est = np.quantile(d_est, 0.9, interpolation='nearest', axis=0)
        stats_inc_orig.loc[i, 'q_p90'] = q_p90
        stats_inc_est.loc[i, 'q_p90'] = q_p90_est
        q_p10 = np.quantile(d, 0.1, interpolation='nearest', axis=0)
        q_p10_est = np.quantile(d_est, 0.1, interpolation='nearest', axis=0)
        stats_inc_orig.loc[i, 'q_p10'] = q_p10
        stats_inc_est.loc[i, 'q_p10'] = q_p10_est
        stats_inc_orig.loc[i, 'tr_range'] = q_p90 - q_p10
        stats_inc_est.loc[i, 'tr_range'] = q_p90_est - q_p10_est
        temp_func = lambda a: np.mean(a[:-2][(a[:-2] >= a[-2]) & (a[:-2] <= a[-1])])
        stats_inc_orig.loc[i, 'tr_mean'] = np.apply_along_axis(temp_func, 0, np.concatenate([d, [q_p10], [q_p90]],
                                                                                            axis=0))
        stats_inc_est.loc[i, 'tr_mean'] = np.apply_along_axis(temp_func, 0, np.concatenate([d_est, [q_p10_est],
                                                                                            [q_p90_est]], axis=0))
        stats_inc_orig.loc[i, 'size'] = s / 4096
        stats_inc_est.loc[i, 'size'] = s / 4096
    centers = pd.DataFrame(centers, columns=pd.MultiIndex.from_product([['center'], ['x', 'y'] + list(channel_keys)]))
    stats_inc_orig = stats_inc_orig.join(centers)
    stats_inc_est = stats_inc_est.join(centers)
    d_base = stats_inc_orig.loc[0, 'q_p10'].to_numpy(dtype=np.float)
    d_Base = stats_inc_orig.loc[int(len(clusters) > 1):, 'q_p90'].to_numpy(dtype=np.float)
    stats = pd.Series(index=pd.MultiIndex.from_product([final_stats_keys, channel_keys]), dtype=np.float)
    div = stats_inc_est.loc[0, 'q_p10'].to_numpy(dtype=np.float)
    # print((stats_inc_est.loc[int(len(clusters) > 1):, 'q_p90'].to_numpy(dtype=np.float) - div) / div)
    # print(d_Base)
    # print(d_base)
    # print((d_Base - d_base) / d_base)
    Ro_cont = ((stats_inc_est.loc[int(len(clusters) > 1):, 'q_p90'].to_numpy(dtype=np.float) - div) / div) / \
              ((d_Base - d_base) / d_base)
    print(div, Ro_cont)
    print(d_Base, d_base)
    print((d_Base - d_base) / d_base)
    Ro_cont[np.isinf(Ro_cont)] = 1
    R_cont = Ro_cont.copy()
    R_cont[R_cont > 1] = 2 - R_cont[R_cont > 1]
    Ro_sep = np.zeros_like(R_cont)
    for i in range(len(Ro_sep)):
        div = np.ones_like(d0) * d_Base[i]
        Ro_sep[i] = 1 - (np.sqrt(mean_squared_error(d0, d0_est, multioutput='raw_values'))
                         / np.sqrt(mean_squared_error(div, d0, multioutput='raw_values')))
    Ro_sep[np.isinf(Ro_sep)] = 1
    # Ro_sep[Ro_sep > 1] = 2 - Ro_sep[Ro_sep > 1]
    sign = np.sign(Ro_sep)
    R_sep = Ro_sep * (1 - sign + sign * R_cont)
    R_sep = np.sign(R_sep) * np.sqrt(np.abs(R_sep))
    Ro_size = np.zeros_like(R_cont)
    for j, i in enumerate(range(int(len(clusters) > 1), len(clusters))):
        di = data_[labels == i]
        di_est = data_est_[labels == i]
        div = np.ones_like(di) * d_base
        Ro_size[j] = 1 - (np.sqrt(mean_squared_error(di, di_est, multioutput='raw_values'))
                          / np.sqrt(mean_squared_error(div, di, multioutput='raw_values')))
    Ro_size[np.isinf(Ro_size)] = 1
    # Ro_size[Ro_size > 1] = 2 - Ro_size[Ro_size > 1]
    sign = np.sign(Ro_size)
    R_size = Ro_size * (1 - sign + sign * R_cont)
    sign = np.sign(R_size)
    R_size = sign * np.sqrt(np.abs(R_size))
    R_csd = R_size * (1 - sign + sign * R_cont)
    R_csd = np.sign(R_csd) * np.sqrt(np.abs(R_csd))
    sz = data.shape[0]
    ssim = np.zeros(sz)
    ssim_grad = np.zeros_like(data)
    ssim_S = np.zeros_like(data)
    for i in range(sz):
        data_range = np.max([data[i], data_est[i]]) - np.min([data[i], data_est[i]])
        ssim[i], ssim_grad[i], ssim_S[i] = structural_similarity(data[i], data_est[i], gradient=True, full=True,
                                                                 data_range=data_range, gaussian_weights=True)
    # print('tes1')
    # print(R_cont)
    # print('tes2')
    stats['R_cont'] = np.mean(R_cont, axis=0)
    stats['Ro_size'] = np.mean(Ro_size, axis=0)
    stats['R_size'] = np.mean(R_size, axis=0)
    stats['Ro_sep'] = np.mean(Ro_sep, axis=0)
    stats['R_sep'] = np.mean(R_sep, axis=0)
    stats['R_csd'] = np.mean(R_csd, axis=0)
    stats['ssim'] = ssim
    image_map = empty_image
    image_map[mask] = labels
    warnings.resetwarnings()
    if plot:
        plt.plot(stats_inc_orig['mean'], stats_inc_est['mean'], '.')
        plt.plot(stats_inc_orig['max'], stats_inc_est['max'], '.')
        plt.plot(stats_inc_orig['min'], stats_inc_est['min'], '.')
        temp = [min(stats_inc_orig['min']), max(stats_inc_orig['max'])]
        plt.plot(temp, temp)
    else:
        return stats, stats_inc_orig, stats_inc_est, image_map, ssim_grad, ssim_S


def ground_truth_vs_estimated_direct(key, MU_est, s_range=slice(None), plot=False):
    s_range = np.arange(len(r_data[key]))[s_range]
    sel_inc = (r_data[key] > 0)[s_range]
    n_inc = np.sum(sel_inc, axis=-1) + 1
    # mu = np.zeros((dataset.MU.shape[1], np.sum(n_inc)))
    # mu_est = np.zeros((dataset.MU.shape[1], np.sum(n_inc)))
    # muM = np.zeros((dataset.MU.shape[1], np.sum(n_inc)))
    # muM_est = np.zeros((dataset.MU.shape[1], np.sum(n_inc)))
    # mum = np.zeros((dataset.MU.shape[1], np.sum(n_inc)))
    # mum_est = np.zeros((dataset.MU.shape[1], np.sum(n_inc)))
    # mus = np.zeros((dataset.MU.shape[1], np.sum(n_inc)))
    # mus_est = np.zeros((dataset.MU.shape[1], np.sum(n_inc)))
    # s_mu = np.zeros((dataset.MU.shape[1], np.sum(n_inc)))
    mu_stats = pd.DataFrame(columns=pd.MultiIndex.from_product([final_stats_keys, channel_keys]))
    mu_stats_inc_orig = pd.DataFrame(columns=pd.MultiIndex.from_product([stats_inc_keys, channel_keys]))
    mu_stats_inc_est = mu_stats_inc_orig.copy()
    mu_image_map = None
    mu_ssim_S = None
    mu_ssim_grad = None
    if plot:
        c_mu = np.zeros((2, np.sum(n_inc)))
        j = 0
    for n, X, X_est, sel_clusters, t_r, t_roc, t_theta, t_ca, t_csp, t_d in zip(s_range, MU[key][s_range],
                                                                                MU_est[s_range],
                                                                                sel_inc, r_data[key][s_range],
                                                                                roc_data[key][s_range],
                                                                                theta_data[key][s_range],
                                                                                ca_data[key][s_range],
                                                                                csp_data[key][s_range],
                                                                                d[key][s_range]):
        centers = np.array([t_roc/(t_d/2), t_theta, t_ca, t_csp])
        centers = centers.reshape(np.flip(centers.shape))[sel_clusters]
        stats, stats_inc_orig, stats_inc_est, image_map, ssim_grad, ssim_S = ground_truth_vs_estimated(
            X, X_est, centers, plot=False)
        mu_stats = mu_stats.append(stats, ignore_index=True)
        mu_stats_inc_orig = mu_stats_inc_orig.append(stats_inc_orig, ignore_index=True)
        mu_stats_inc_est = mu_stats_inc_est.append(stats_inc_est, ignore_index=True)
        if mu_image_map is None:
            mu_image_map = np.array([image_map])
        else:
            mu_image_map = np.concatenate([mu_image_map, [image_map]], axis=0)
        if mu_ssim_S is None:
            mu_ssim_S = np.array([ssim_S])
        else:
            mu_ssim_S = np.concatenate([mu_ssim_S, [ssim_S]], axis=0)
        if mu_ssim_grad is None:
            mu_ssim_grad = np.array([ssim_grad])
        else:
            mu_ssim_grad = np.concatenate([mu_ssim_grad, [ssim_grad]], axis=0)
        if plot:
            sz = len(stats_inc_orig)
            c = t_r + t_roc
            ind = np.flip(np.argsort(c))
            c = c[ind]
            temp = t_ca[ind]
            c_mu[0, j:(j + sz)] = c[temp > 1][0]
            temp = t_csp[ind]
            c_mu[1, j:(j + sz)] = c[temp > 1][0]
            j += sz
    if plot:
        # mu = np.delete(mu, slice(j, np.sum(n_inc), 1), axis=1)
        # mu_est = np.delete(mu_est, slice(j, np.sum(n_inc), 1), axis=1)
        # muM = np.delete(muM, slice(j, np.sum(n_inc), 1), axis=1)
        # muM_est = np.delete(muM_est, slice(j, np.sum(n_inc), 1), axis=1)
        # mum = np.delete(mum, slice(j, np.sum(n_inc), 1), axis=1)
        # mum_est = np.delete(mum_est, slice(j, np.sum(n_inc), 1), axis=1)
        # mus = np.delete(mus, slice(j, np.sum(n_inc), 1), axis=1)
        # mus_est = np.delete(mus_est, slice(j, np.sum(n_inc), 1), axis=1)
        # s_mu = np.delete(s_mu, slice(j, np.sum(n_inc), 1), axis=1)
        sz = len(mu_stats_inc_orig)
        c_mu = np.delete(c_mu, slice(j, np.sum(n_inc), 1), axis=1)
        alpha_mu = 1 / (sz * 5 / 25400 + 5)
        _, axes = plt.subplots(nrows=1, ncols=2)
        reg = LinearRegression()
        for i in range(2):
            l = np.array([[np.min(mu_stats_inc_orig['min'][i])], [np.max(mu_stats_inc_orig['max'][i])]])
            ym = l[0, 0]
            yM = l[1, 0]
            reg.fit(mu_stats_inc_orig['max'][i].reshape(-1, 1), mu_stats_inc_est['max'][i].reshape(-1, 1))
            lrM = reg.predict(l)
            yM = max(yM, np.max(lrM))
            reg.fit(mu_stats_inc_orig['min'][i].reshape(-1, 1), mu_stats_inc_est['min'][i].reshape(-1, 1))
            lrm = reg.predict(l)
            ym = min(ym, np.min(lrm))
            yM2 = np.max(mu_stats_inc_est['max'][i])
            ym2 = np.min(mu_stats_inc_est['min'][i])
            b = False
            if ym2 < 2 * ym - yM:
                axes[i].set_ylim(bottom=2 * ym - yM)
                b = True
            if yM2 > 2 * yM - ym:
                axes[i].set_ylim(top=2 * yM - ym)
                b = True
            if b:
                axes[i].text(0, 1, 'min = ' + str(ym2) + '; max = ' + str(yM2), horizontalalignment='left',
                             verticalalignment='bottom', transform=axes[i].transAxes)
            axes[i].scatter(mu_stats_inc_orig['max'][i], mu_stats_inc_est['max'][i],
                            s=mu_stats_inc_orig['size'][i] * 40, alpha=alpha_mu,
                            c=c_mu[i], marker='^', cmap='Oranges')
            axes[i].plot(l, lrM, c='C1')
            axes[i].scatter(mu_stats_inc_orig['min'][i], mu_stats_inc_est['min'][i],
                            s=mu_stats_inc_orig['size'][i] * 40, alpha=alpha_mu,
                            c=c_mu[i], marker='v', cmap='Greens')
            axes[i].plot(l, lrm, c='C2')
            # axes[i].scatter(mu[i] + mus[i], mu_est[i] + mus_est[i], s=s_mu[i] * 40 / 4096, alpha=alpha_mu,
            #                 c=c_mu[i], marker='2', cmap='Reds')
            # reg.fit((mu[i] + mus[i]).reshape(-1, 1), (mu_est[i] + mus_est[i]).reshape(-1, 1))
            # axes[i].plot(l, reg.predict(l), c='C3')
            # axes[i].scatter(mu[i] - mus[i], mu_est[i] - mus_est[i], s=s_mu[i] * 40 / 4096, alpha=alpha_mu,
            #                 c=c_mu[i], marker='1', cmap='Purples')
            # reg.fit((mu[i] - mus[i]).reshape(-1, 1), (mu_est[i] - mus_est[i]).reshape(-1, 1))
            # axes[i].plot(l, reg.predict(l), c='C4')
            axes[i].scatter(mu_stats_inc_orig['mean'][i], mu_stats_inc_est['mean'][i],
                            s=mu_stats_inc_orig['size'][i] * 40,
                            alpha=alpha_mu, c=c_mu[i], cmap='Blues')
            reg.fit(mu_stats_inc_orig['mean'][i].reshape(-1, 1), mu_stats_inc_est['mean'][i].reshape(-1, 1))
            axes[i].plot(l, reg.predict(l), c='C0')
            axes[i].plot(l, l, 'C5')
    return mu_stats, mu_stats_inc_orig, mu_stats_inc_est, mu_image_map, mu_ssim_grad, mu_ssim_S


def save_results(filename):
    savemat(filename, MU_deep)


def load_results(filename):
    global MU_deep
    MU_deep = loadmat(filename)


def plot_result(s_num, dataset_name='training', color_range='normal', vmin=None, vmax=None,
                roc=None, roc_index=None, correction=True, contrast=False, title=None):
    if correction:
        MU_iter_ = MU_iter_corrected
    else:
        MU_iter_ = MU_iter_uncorrected
    if not isinstance(vmin, list) or len(vmin) != 2:
        vmin = [None, None]
    if not isinstance(vmax, list) or len(vmax) != 2:
        vmax = [None, None]
    if title is None:
        title = ''
    title = str(title)
    scale = d[dataset_name][s_num] / 2
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    plt.rcParams.update({'font.size': 18})
    fig.suptitle(title)
    plt.rcParams.update({'font.size': 10})
    axes = axes.flatten()
    for ax in axes:
        ax.set_aspect('equal')
    truth_image = MU[dataset_name][s_num]
    standard_image = MU_iter_[dataset_name][s_num]
    try:
        deep_image = MU_deep[dataset_name][s_num]
    except:
        temp = mask * np.nan
        deep_image = np.concatenate([[temp], [temp]])
    # truth_title = ['ground truth (MUa)', 'ground truth (MUsp)']
    # standard_title = ['Tikhonov reg. method (MUa)', 'Tikhonov reg. method (MUsp)']
    # deep_title = ['deep neural networks (MUa)', 'deep neural networks (MUsp)']
    if contrast:
        MU0_ = np.reshape(MU0[dataset_name][s_num], MU0[dataset_name][s_num].shape + (1, 1))
        MU0_deep_ = np.reshape(MU0_deep[dataset_name][s_num], MU0_deep[dataset_name][s_num].shape + (1, 1))
        truth_image /= MU0_
        standard_image /= MU0_
        deep_image /= MU0_deep_
        # truth_title = [s1 + ' homo=' + str(s2) for s1, s2 in zip(['MUa', 'MUsp'], MU0_.flatten())]
        # standard_title = [s1 + ' homo=' + str(s2) for s1, s2 in zip(['MUa', 'MUsp'], MU0_.flatten())]
        # deep_title = [s1 + ' homo=' + str(s2) for s1, s2 in zip(['MUa', 'MUsp'], MU0_deep_.flatten())]
    images = [axes[0].pcolormesh(X2 * scale, Y2 * scale, mask_image(truth_image[0], np.nan), cmap='inferno'),
              axes[1].pcolormesh(X2 * scale, Y2 * scale, mask_image(standard_image[0], np.nan), cmap='inferno'),
              axes[2].pcolormesh(X2 * scale, Y2 * scale, mask_image(deep_image[0], np.nan), cmap='inferno')]
    # axes[0].set_title(truth_title[0])
    # axes[1].set_title(standard_title[0])
    # axes[2].set_title(deep_title[0])
    plt.rcParams.update({'font.size': 8})
    axes[0].set_title('ground truth')
    axes[1].set_title('Tikhonov reg. method')
    axes[2].set_title('deep neural networks')
    plt.rcParams.update({'font.size': 10})
    axes[0].set_ylabel(r'$\mu_a$', fontsize=18)
    if color_range == 'normal':
        vmin = [0, 0]
        vmax = [np.max(mask_image(truth_image[i], -np.inf)) * 2 for i in range(2)]
    elif color_range == 'auto':
        vmin = [min(np.min(mask_image(truth_image[i], np.inf)), np.min(mask_image(standard_image[i], np.inf)),
                    np.min(mask_image(deep_image[i], np.inf))) for i in range(2)]
        vmax = [max(np.max(mask_image(truth_image[i], -np.inf)), np.max(mask_image(standard_image[i], -np.inf)),
                    np.max(mask_image(deep_image[i], -np.inf))) for i in range(2)]
    if color_range is None:
        fig.colorbar(images[0], ax=axes[0], orientation='horizontal', fraction=.1)
        fig.colorbar(images[1], ax=axes[1], orientation='horizontal', fraction=.1)
        fig.colorbar(images[2], ax=axes[2], orientation='horizontal', fraction=.1)
    else:
        for im in images:
            im.set_clim(vmin=vmin[0], vmax=vmax[0])
        fig.colorbar(images[0], ax=axes[0:3], orientation='horizontal', fraction=.1)
    images = [axes[3].pcolormesh(X2 * scale, Y2 * scale, mask_image(truth_image[1], np.nan), cmap='jet'),
              axes[4].pcolormesh(X2 * scale, Y2 * scale, mask_image(standard_image[1], np.nan),
                                 cmap='jet'),
              axes[5].pcolormesh(X2 * scale, Y2 * scale, mask_image(deep_image[1], np.nan),
                                 cmap='jet')]
    # axes[3].set_title(truth_title[1])
    # axes[4].set_title(standard_title[1])
    # axes[5].set_title(deep_title[1])
    axes[3].set_ylabel(r'$\mu^\prime_s$', fontsize=18)
    if color_range is None:
        fig.colorbar(images[0], ax=axes[3], orientation='horizontal', fraction=.1)
        fig.colorbar(images[1], ax=axes[4], orientation='horizontal', fraction=.1)
        fig.colorbar(images[2], ax=axes[5], orientation='horizontal', fraction=.1)
    else:
        for im in images:
            im.set_clim(vmin=vmin[1], vmax=vmax[1])
        fig.colorbar(images[0], ax=axes[3:6], orientation='horizontal', fraction=.1)
    if roc_index is not None:
        roc = roc_data[dataset_name][s_num, roc_index]
    if roc is not None:
        for ax in axes[:3]:
            ax.plot(roc * np.cos(p_theta), roc * np.sin(p_theta), c='#00ff00')
        for ax in axes[3:]:
            ax.plot(roc * np.cos(p_theta), roc * np.sin(p_theta), c='#000000')


def plot_result_profile(s_num, dataset_name='training', roc=None, roc_index=0, correction=True, title=None):
    if correction:
        MU_iter_ = MU_iter_corrected
    else:
        MU_iter_ = MU_iter_uncorrected
    if roc is None:
        roc = roc_data[dataset_name][s_num, roc_index]
    if title is None:
        title = ''
    title = str(title)
    scale = d[dataset_name][s_num] / 2
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(nrows=2, sharex=True)
    plt.rcParams.update({'font.size': 18})
    fig.suptitle(title)
    plt.rcParams.update({'font.size': 10})
    for i in range(2):
        points = (X.flatten() * scale, Y.flatten() * scale)
        axes[i].plot(p_theta * 180 / np.pi, griddata(points, MU[dataset_name][s_num, i].flatten(),
                                                     (roc * np.cos(p_theta), roc * np.sin(p_theta))))
        axes[i].plot(p_theta * 180 / np.pi, griddata(points, MU_iter_[dataset_name][s_num, i].flatten(),
                                                     (roc * np.cos(p_theta), roc * np.sin(p_theta))))
        axes[i].plot(p_theta * 180 / np.pi, griddata(points, MU_deep[dataset_name][s_num, i].flatten(),
                                                     (roc * np.cos(p_theta), roc * np.sin(p_theta))))
    axes[0].set_ylabel(r'$\mu_a$ $(mm^{-1})$')
    axes[1].set_ylabel(r'$\mu^\prime_s$ $(mm^{-1})$')
    axes[1].set_xlabel('angular distance (degree)')
    axes[0].set_title('radius %g mm' % roc, fontsize=8)
    axes[0].legend(['ground truth', 'Tikhonov reg. method', 'deep neural networks'], loc='upper right')
    return roc


def fit_model(model_input, in_train, out_train, validation_data=None, batch_size=32, epochs=100, verbose=1,
              callbacks=None, format_string_batch=None, loss_index_batch=None, format_string_epoch=None,
              loss_index_epoch=None, apply_noise=False):
    logger = BaseLogger(stateful_metrics=model_input.metrics_names[1:])
    history = History()
    callbacks = [logger, history] + callbacks
    if isinstance(in_train, list):
        t_size = len(in_train[0])
        meas = in_train[0]
    else:
        t_size = len(in_train)
        meas = in_train
    out_labels = model_input.metrics_names
    callback_metrics = copy.copy(out_labels)
    if validation_data is not None:
        if not isinstance(validation_data[0], tuple) and not isinstance(validation_data[0], list):
            validation_data = [validation_data]
        for i in range(len(validation_data)):
            callback_metrics += ['val_' + str(i) + '_' + s for s in out_labels]
    callbacks = CallbackList(callbacks)
    callbacks.set_params({
        'batch_size': batch_size,
        'epochs': epochs,
        'steps': None,
        'samples': t_size,
        'verbose': verbose,
        'do_validation': validation_data is not None,
        'metrics': callback_metrics,
    })
    callbacks.on_train_begin()
    if PREPROCESSING_MODE == 'FIXED':
        seq_len_train = np.ones(t_size, dtype=np.int) * meas.shape[1]
    else:
        seq_len_train = np.sum(meas[:, :, -1], axis=1).astype(np.int)
    index_array = np.argsort(seq_len_train)[::-1]
    nmeas_u, idx = np.unique(seq_len_train, return_inverse=True)
    idx = idx[index_array]
    meas_used = meas
    # if shuffle_meas:
    #     PHI_meas_train = PHI_meas_train.copy()
    #     timesteps_array = np.arange(PHI_meas_train.shape[1])
    num_batches = (t_size + batch_size - 1) // batch_size
    for epoch in range(epochs):
        model_input.reset_metrics()
        callbacks.on_epoch_begin(epoch)
        epoch_logs = {}
        progress_bar = None
        for u in nmeas_u:
            np.random.shuffle(index_array[nmeas_u[idx] == u])
        if apply_noise:
            meas_used = meas.copy()
            raw = get_raw_measurements(meas_used)
            noise_flag = np.random.random((t_size, 1)) >= 0.5
            meas_noise = np.random.random((t_size, 1)) * noise_max * noise_flag
            meas_add_noise = np.random.random((t_size, 1)) * noise_add_max * noise_flag
            noise = np.apply_along_axis(lambda a: (np.random.normal(scale=1 / np.sqrt(2), size=raw.shape[1:]) +
                                                   np.random.normal(scale=1 / np.sqrt(2), size=raw.shape[1:]) * 1j) * a,
                                        1, meas_noise)
            raw = raw * (1 + noise)
            noise = np.apply_along_axis(lambda a: (np.random.normal(scale=1 / np.sqrt(2), size=raw.shape[1:]) +
                                                   np.random.normal(scale=1 / np.sqrt(2), size=raw.shape[1:]) * 1j) * a,
                                        1, meas_add_noise)
            raw = raw + np.std(raw, axis=(1, 2)).reshape((-1, 1, 1)) * noise
            meas_used = set_raw_measurements(meas_used, raw)
        batch_end = 0
        # if shuffle_meas and epoch % shuffle_period == 0:
        #     for x in PHI_meas_train:
        #         np.random.shuffle(x)
        seq_sel = np.random.random((num_batches, meas_used.shape[1])) >= 0.3
        for batch_index in range(num_batches):
            batch_start = batch_end
            batch_end = min(t_size, batch_end + batch_size)
            batch_ids = index_array[batch_start:batch_end]
            batch_seq_len = seq_len_train[batch_ids]
            batch_logs = {'batch': batch_index, 'size': len(batch_ids)}
            callbacks.on_batch_begin(batch_index, batch_logs)
            max_seq_len = max(batch_seq_len)
            if PREPROCESSING_MODE != 'FIXED':
                sel_prob = np.random.random() >= 0.5
                if sel_prob:
                    batch_seq_sel = seq_sel[batch_index]
                    batch_seq_sel[max_seq_len:] = 0
                    batch_in_train = meas_used[batch_ids][:, batch_seq_sel]
                else:
                    batch_in_train = meas_used[batch_ids][:, :max_seq_len]
            else:
                batch_in_train = meas_used[batch_ids]
            if isinstance(in_train, list):
                batch_in_train = [batch_in_train] + [x[batch_ids] for x in in_train[1:]]
            # if shuffle_meas:
            #     np.random.shuffle(timesteps_array)
            #     for i in range(len(batch_in_train)):
            #         batch_in_train[i] = batch_in_train[i, timesteps_array]
            if isinstance(out_train, list):
                batch_out_train = [x[batch_ids] for x in out_train]
            else:
                batch_out_train = out_train[batch_ids]
            loss = model_input.train_on_batch(batch_in_train, batch_out_train)
            for l, o in zip(out_labels, loss):
                batch_logs[l] = o
            callbacks.on_batch_end(batch_index, batch_logs)
            if verbose:
                if progress_bar is None:
                    progress_bar = tqdm(total=t_size, desc='Epoch {}/{}'.format(epoch + 1, epochs), unit='samples')
                if format_string_batch is not None and loss_index_batch is not None:
                    progress_bar.set_postfix(train_on_batch=format_string_batch % tuple(loss[i]
                                                                                        for i in loss_index_batch))
                progress_bar.update(len(batch_ids))
        if validation_data is not None:
            if verbose:
                progress_bar.set_postfix_str('validating..')
            for i, v in enumerate(validation_data):
                in_val, out_val = v
                val_loss = model_input.evaluate(in_val, out_val, batch_size=batch_size, verbose=0)
                for l, o in zip(out_labels, val_loss):
                    epoch_logs['val_' + str(i) + '_' + l] = o
        callbacks.on_epoch_end(epoch, epoch_logs)
        if verbose:
            if format_string_epoch is not None and loss_index_epoch is not None:
                progress_bar.set_postfix(train=format_string_epoch % tuple(epoch_logs[out_labels[i]]
                                                                           for i in loss_index_epoch),
                                         validation=[format_string_epoch %
                                                     tuple(epoch_logs['val_' + str(j) + '_' + out_labels[i]]
                                                           for i in loss_index_epoch)
                                                     for j in range(len(validation_data))])
            else:
                progress_bar.set_postfix_str('')
            progress_bar.close()
    callbacks.on_train_end()
    return history


def plot_history(history, indices=(None, None), add_label_indices=(0,), subtract_label_indices=None):
    if isinstance(add_label_indices, int):
        add_label_indices = (add_label_indices,)
    if subtract_label_indices is None:
        subtract_label_indices = ()
    if isinstance(subtract_label_indices, int):
        subtract_label_indices = (subtract_label_indices,)
    keys = []
    for k in history.keys():
        if not k.startswith('val_'):
            keys.append(k)
    if len(add_label_indices + subtract_label_indices) == 1:
        label = keys[add_label_indices[0]]
        val_label = 'val_' + label
    else:
        label = 'training'
        val_label = 'validation'
    data = history[keys[add_label_indices[0]]]
    val_data = history['val_' + keys[add_label_indices[0]]]
    title = keys[add_label_indices[0]]
    for i in add_label_indices[1:]:
        data += history[keys[i]]
        val_data += history['val_' + keys[i]]
        title += ' + ' + keys[i]
    for i in subtract_label_indices:
        data -= history[keys[i]]
        title += ' - ' + keys[i]
    plt.figure()
    plt.plot(data[indices[0]:indices[1]])
    plt.plot(val_data[indices[0]:indices[1]])
    plt.title(title)
    plt.ylabel('metric')
    plt.xlabel('epoch')
    plt.legend([label, val_label], loc='upper left')
    plt.show()

# set_model('model9log.h5', PHI_meas)
