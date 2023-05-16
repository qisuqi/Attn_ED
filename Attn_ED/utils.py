import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
import seaborn as sns
from itertools import cycle


def remove_static_cols(col_names, timesteps=3):

    t_0 = [f'{x}_t0' for x in col_names]
    t_0[0] = 'ID'
    t_0[1] = 'Date'
    t_0[2] = 'M'
    t_0[3] = 'F'
    t_0[4] = 'Age'

    t_1 = [f'{x}_t1' for x in col_names]
    t_1.remove('ID_t1')
    t_1.remove('Date_t1')
    t_1.remove('M_t1')
    t_1.remove('F_t1')
    t_1.remove('Age_t1')

    t_2 = [f'{x}_t2' for x in col_names]
    t_2.remove('ID_t2')
    t_2.remove('Date_t2')
    t_2.remove('M_t2')
    t_2.remove('F_t2')
    t_2.remove('Age_t2')

    t_3 = [f'{x}_t3' for x in col_names]
    t_3.remove('ID_t3')
    t_3.remove('Date_t3')
    t_3.remove('M_t3')
    t_3.remove('F_t3')
    t_3.remove('Age_t3')

    if timesteps == 2:
        column_names = sum([t_0, t_1], [])
    elif timesteps == 3:
        column_names = sum([t_0, t_1, t_2], [])
    else:
        raise Exception('Too many timesteps')
    return column_names


def compute_mape(true, pred):

    eps = np.finfo(np.float64).eps

    return np.mean(np.abs(true - pred) / np.maximum(np.abs(true), eps))


def compute_wape(true, pred):

    nominator = np.sum(np.abs(true - pred))
    denominator = np.sum(np.abs(true))

    return nominator / denominator


def compute_mase(train, true, pred):
    pred_naive = []
    for i in range(1, len(train)):
        pred_naive.append(train[(i - 1)])

    mae_naive = np.mean(abs(train[1:] - pred_naive))

    return np.mean(abs(true - pred)) / mae_naive


def compute_sampe(true, pred):

    nominator = np.abs(true - pred)
    denominator = np.abs(true) + np.abs(pred)

    return np.mean(2.0 * nominator / denominator)


def error_estimator(true, pred, train):
    print('-----------------------------')
    smape = compute_sampe(true, pred)
    print("sMAPE is:", smape)
    mase = compute_mase(train, true, pred)
    print("MASE is:",  mase)
    mape = compute_mape(true, pred)
    print('MAPE is:', mape)
    wape = compute_wape(true, pred)
    print('WAPE is', wape)
    print('-----------------------------')


def prepare_targets(y):
    le = LabelEncoder()
    le.fit(y)
    y_enc = le.transform(y)
    return y_enc


def plot_loss(loss, val_loss):
    plt.plot(loss, label='Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Model Evaluation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


def plot_acc(accuracy, val_accuracy):
    plt.plot(accuracy, label='Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.legend()
    plt.title('Model Evaluation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


def plot_rmse(rmse, val_rmse):
    plt.plot(rmse, label='RMSE')
    plt.plot(val_rmse, label='Validation RMSE')
    plt.legend()
    plt.title('Model Evaluation')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.show()


def roc_auc_score_multiclass(actual_class, pred_class, average="macro", return_type='auc'):
    # creating a set of all the unique classes using the actual class list
    unique_class = actual_class.unique()
    roc_auc_dict = {}

    for per_class in unique_class:
        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        # using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average=average)

        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict


def plot_roc(fpr, tpr, roc_auc):

    labels = list(fpr.keys())

    fig, ax = plt.subplots()
    ax.plot(fpr[labels[0]],
             tpr[labels[0]],
             color='aqua',
             label='ROC curve (area = %0.2f)' % roc_auc[labels[0]])
    ax.plot(fpr[labels[1]],
             tpr[labels[1]],
             color='darkorange',
             label='ROC curve (area = %0.2f)' % roc_auc[labels[1]])
    ax.plot(fpr[labels[2]],
             tpr[labels[2]],
             color='cornflowerblue',
             label='ROC curve (area = %0.2f)' % roc_auc[labels[2]])
    ax.plot([0, 1], [0, 1], color='navy', linestyle='--')

    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Testing ROC Curves')
    plt.legend(loc="lower right")
    plt.show()


def plot_results(date, true, pred):
    plt.plot(date, true, '.-', label='Actual')
    plt.plot(date, pred, '.--', label='Prediction')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.title('Prediction Results')
    plt.tight_layout()
    plt.show()


def preprocess_multivariate_ts(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-d%d)' % (j+1, i)) for j in range(n_vars)]

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    aggs = pd.concat(cols, axis=1)
    aggs.columns = names

    if dropnan:
        aggs.dropna(inplace=True)
    return aggs


def compute_usage_intervals(ts, d_max):
    ts = pd.to_datetime(ts, errors='coerce')
    ts_len = len(ts)
    d_max = int(d_max)
    current_interval = 0
    intervals = []
    intervals.append([])
    durations = []

    for i in range(0, ts_len - 1):

        distance = abs((ts[i + 1] - ts[i]).total_seconds())

        if distance <= d_max:
            intervals[current_interval].append(ts[i + 1])
        else:
            current_interval += 1
            intervals.append([])
            intervals[current_interval].append(ts[i + 1])

    intervals[0].insert(0, ts[0])

    for date in intervals:
        dr = (date[-1] - date[0]).total_seconds()
        durations.append((date[0].strftime('%Y-%m-%d'), dr))

    return durations


def inverse_transform(scaler, normaliser, array, dim=2):
    if dim == 2:
        inverse_array = normaliser.inverse_transform(array)
        inverse_array = scaler.inverse_transform(inverse_array)
    else:
        reshape_array = np.reshape(array, (array.shape[0], array.shape[1]))
        inversed = normaliser.inverse_transform(reshape_array)
        inversed = scaler.inverse_transform(inversed)
        inverse_array = np.reshape(inversed, (inversed.shape[0], inversed.shape[1], 1))
    return inverse_array


def get_class_weights(data):

    one, zero = np.bincount(data['Outcome'])
    total = one + zero

    weight_for_0 = (1 / one) * (total / 2.0)
    weight_for_1 = (1 / zero) * (total / 2.0)

    initial_bias = np.log([zero / one]).item()

    initial_bias = tf.keras.initializers.Constant(initial_bias)

    return weight_for_0, weight_for_1, initial_bias



