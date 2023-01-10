from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split


def remove_static_cols(col_names, timesteps=3):

    t_0 = [f'{x}_t0' for x in col_names]
    t_0[0] = 'ID'
    t_0[1] = 'Date'
    t_0[2] = 'Sex_1'
    t_0[3] = 'Sex_2'
    t_0[4] = 'Age'

    t_1 = [f'{x}_t1' for x in col_names]
    t_1.remove('ID_t1')
    t_1.remove('Date_t1')
    t_1.remove('Sex_1_t1')
    t_1.remove('Sex_2_t1')
    t_1.remove('Age_t1')

    t_2 = [f'{x}_t2' for x in col_names]
    t_2.remove('ID_t2')
    t_2.remove('Date_t2')
    t_2.remove('Sex_1_t2')
    t_2.remove('Sex_2_t2')
    t_2.remove('Age_t2')

    t_3 = [f'{x}_t3' for x in col_names]
    t_3.remove('ID_t3')
    t_3.remove('Date_t3')
    t_3.remove('Sex_1_t3')
    t_3.remove('Sex_2_t3')
    t_3.remove('Age_t3')

    t_4 = [f'{x}_t4' for x in col_names]
    t_4.remove('ID_t4')
    t_4.remove('Date_t4')
    t_4.remove('Sex_1_t4')
    t_4.remove('Sex_2_t4')
    t_4.remove('Age_t4')

    t_5 = [f'{x}_t5' for x in col_names]
    t_5.remove('ID_t5')
    t_5.remove('Date_t5')
    t_5.remove('Sex_1_t5')
    t_5.remove('Sex_2_t5')
    t_5.remove('Age_t5')

    t_6 = [f'{x}_t6' for x in col_names]
    t_6.remove('ID_t6')
    t_6.remove('Date_t6')
    t_6.remove('Sex_1_t6')
    t_6.remove('Sex_2_t6')
    t_6.remove('Age_t6')

    if timesteps == 2:
        column_names = sum([t_0, t_1], [])
    elif timesteps == 3:
        column_names = sum([t_0, t_1, t_2], [])
    else:
        raise Exception('Too many timesteps')
    return column_names


def compute_mape(true, pred):
    if true.size != pred.size:
        print("Number of true and pred do not match")

    mape = np.mean((np.abs((true - pred)/true)))
    return mape


def compute_wape(true, pred):
    if true.size != pred.size:
        print("Number of true and pred do not match")

    nume = np.sum(np.abs(true - pred))
    denom = np.sum(np.abs(true))
    wape = nume / denom
    return wape


def compute_mase(train, true, pred):
    pred_naive = []
    for i in range(1, len(train)):
        pred_naive.append(train[(i - 1)])

    mase = np.mean(abs(train[1:] - pred_naive))

    return np.mean(abs(true - pred)) / mase


def compute_sampe(true, pred):
    true = np.reshape(true, (-1, ))
    pred = np.reshape(pred, (-1, ))

    return np.mean(2.0 * np.abs(true - pred) / (np.abs(true) + np.mean(pred))).item()


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


def plot_roc(labels, predictions):
    fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
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

