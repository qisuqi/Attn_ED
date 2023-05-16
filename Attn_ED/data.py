import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder
from scipy import stats
import warnings
from AttnED_TensorFlow import utils


class ProcessData:
    def __init__(self):

        self.normaliser = MinMaxScaler()
        self.scaler = StandardScaler()
        self.ordinal = OrdinalEncoder()
        self.onehot = OneHotEncoder()

    def to_df(self, data):
        return pd.DataFrame(data)

    def compute_usage(self, data, ids, date):
        """ Compute usage interval, by taking the difference between two consecutive timestamps. If the difference
         between two timestamps are greater than 600 seconds, then it is inactive. """

        data = self.to_df(data)
        data.sort_values(by=date, inplace=True)
        data['Date'] = pd.to_datetime(data[date], errors='coerce').dt.strftime('%Y-%m-%d')

        usage_interval = []
        for i in data[ids].unique():
            sub_data = data[data[ids] == i]
            sub_data.reset_index(inplace=True)
            ui = utils.compute_usage_intervals(sub_data[date], 600)
            ui_id = [i, ui]
            usage_interval.append(ui_id)

        flat_usage_interval = []
        for x, y in usage_interval:
            for i in y:
                array = (x, i)
                flat_usage_interval.append(array)

        flat_usage_interval_df = pd.DataFrame(flat_usage_interval, columns=['ID', 'Info'])

        df1 = pd.DataFrame(flat_usage_interval_df.ID.values.tolist(), columns=['ID'])
        df2 = pd.DataFrame(flat_usage_interval_df.Info.values.tolist(), columns=['Date', 'Usage'])

        usage_interval_df = pd.concat([df1, df2], axis=1)
        usage_interval_df.Date = usage_interval_df.Date.astype('datetime64[ns]')

        # Sum up the usage per day.
        usage_sum = []
        for j in usage_interval_df['ID'].unique():
            sub_data = usage_interval_df[usage_interval_df['ID'] == j]
            sub_data = sub_data.set_index('Date').sort_index()
            new_data = sub_data.groupby('Date').sum()
            new_data['Date'] = new_data.index
            array = [j, np.array(new_data)]
            usage_sum.append(array)

        flat_usage_sum = []
        for x, y in usage_sum:
            for i in y:
                array = [x, i]
                flat_usage_sum.append(array)

        flat_usage_sum_df = pd.DataFrame(flat_usage_sum, columns=['ID', 'Info'])

        df3 = pd.DataFrame(flat_usage_sum_df.ID.values.tolist(), columns=['ID'])
        df4 = pd.DataFrame(flat_usage_sum_df.Info.values.tolist(), columns=['ID1', 'Usage', 'Date'])

        all_usage = pd.concat([df3, df4], axis=1)
        all_usage = all_usage.drop(['ID1'], axis=1)
        all_usage = all_usage[all_usage['Usage'] != 0]

        all_usage.Date = all_usage.Date.astype('datetime64[ns]')
        data.Date = data.Date.astype('datetime64[ns]')

        all_usage_df = pd.merge(data, all_usage, on=['ID', 'Date'])
        all_usage_df = all_usage_df.drop([date], axis=1)

        return all_usage_df

    def missing_variables(self, data, ids, date):
        """ Handle missing data. The only variable with missing data here is hVol, so it is imputed based on hProg.
                Evotion methods will be introduced later. """

        warnings.filterwarnings("ignore")

        data = self.to_df(data)
        data = self.compute_usage(data, ids, date)

        del_na = []
        for i in data['ID'].unique():
            sub_data = data[data['ID'] == i]
            sub_data['hVol'] = sub_data['hVol'].fillna(sub_data['hVol'].mean())
            sub_data['LonRel'] = sub_data['LonRel'].fillna(sub_data['LonRel'].mean())
            sub_data['LatRel'] = sub_data['LatRel'].fillna(sub_data['LatRel'].mean())
            array = np.array(sub_data)
            del_na.append(array)

        flat_del_na = []
        for x in del_na:
            for y in x:
                flat_del_na.append(y)

        del_na_df = pd.DataFrame(flat_del_na, columns=data.columns)
        del_na_df.head()

        return del_na_df

    def treat_cat_vars(self, data, ordinal_col, nominal_col):
        """ Pre-processing categorical variables. Ordinal variables are transformed with ordinal encoding and nominal
        variables are transformed with one-hot encoding."""

        data = self.to_df(data)
        #data = self.compute_usage(data, ids, date)

        if len(ordinal_col) == 0 or len(nominal_col) == 0:
            pass

        for i in ordinal_col:
            col_names = f'{i}_mapped'
            data[col_names] = self.ordinal.fit_transform(np.array(data[i]).reshape(-1, 1))
            data = data.drop([i], axis=1)
            data = data.rename(columns={col_names: i})

        for i in nominal_col:
            col_name1, col_name2 = f'{i}_1', f'{i}_2'
            onehot_encoded = self.onehot.fit_transform(np.array(data[i]).reshape(-1, 1)).toarray()
            data = data.drop([i], axis=1)
            data[col_name1] = onehot_encoded[:, 0]
            data[col_name2] = onehot_encoded[:, 1]

        return data

    def aggregate_variables(self, data, groupby, sec_groupby=None):
        """ Aggregate the variables by the taking the average. """

        #data = self.treat_cat_vars(ids, date, ordinal_col, nominal_col)
        data = self.to_df(data)

        average = []
        for i in data[groupby].unique():
            sub_data = data[data[groupby] == i]
            if sec_groupby:
                for j in sub_data[sec_groupby].unique():
                    sub_sub_data = sub_data[sub_data[sec_groupby] == j]
                    grouped_data = sub_sub_data.groupby('Date').mean()
                    grouped_data['Date'] = grouped_data.index
                    array = [i, j, np.array(grouped_data)]
                    average.append(array)
            else:
                grouped_data = sub_data.groupby('Date').mean()
                grouped_data['Date'] = grouped_data.index
                array = [i, np.array(grouped_data)]
                average.append(array)

        flat_average = []

        if sec_groupby:
            for x, y, z in average:
                for i in z:
                    array = [x, y, i]
                    flat_average.append(array)
            average_df = pd.DataFrame(flat_average, columns=[groupby, sec_groupby, 'Info'])
        else:
            for x, y in average:
                for i in y:
                    array = [x, i]
                    flat_average.append(array)

            average_df = pd.DataFrame(flat_average, columns=[groupby, 'Info'])

        col_names = list(data.columns)
        col_names[0] = 'ID1'
        col_names.pop(col_names.index('Date'))
        col_names.append('Date')

        df5 = pd.DataFrame(average_df.ID.values.tolist(), columns=['ID'])
        df6 = pd.DataFrame(average_df.Info.values.tolist(), columns=col_names)

        final_df = pd.concat([df5, df6], axis=1)
        final_df = final_df.drop(['ID1'], axis=1)

        return final_df

    def missing_days(self, data, weekday=True):

        #data = self.aggregate_variables(ids, date, ordinal_col, nominal_col, groupby, sec_groupby)
        data = self.to_df(data)

        cols = list(data.columns)
        cols.remove('Date')

        missing_days = []
        for i in data['ID'].unique():
            sub_data = data[data['ID'] == i]
            idx = pd.date_range(sub_data['Date'].iloc[0], sub_data['Date'].iloc[-1])
            sub_data = sub_data.set_index('Date')
            reindx_subdata = sub_data.reindex(idx)
            reindx_subdata['Usage'] = reindx_subdata['Usage'].interpolate(method='linear')
            #reindx_subdata['Usage'] = reindx_subdata['Usage'].fillna(reindx_subdata['Usage'].mean())
            for j in cols:
                reindx_subdata[j] = reindx_subdata[j].fillna(reindx_subdata[j].mean())
            if weekday:
                reindx_subdata['Weekday'] = pd.to_datetime(reindx_subdata.index).weekday
                reindx_subdata['Weekday_mapped'] = self.ordinal.fit_transform(np.array(reindx_subdata['Weekday']).
                                                                              reshape(-1, 1))
                reindx_subdata = reindx_subdata.drop(['Weekday'], axis=1)
                reindx_subdata = reindx_subdata.rename(columns={'Weekday_mapped': 'Weekday'})
            reindx_subdata['Date'] = reindx_subdata.index
            reindx_subdata.reset_index()
            missing_days.append(np.array(reindx_subdata))

        flat_missing_days = []
        for x in missing_days:
            for y in x:
                flat_missing_days.append(y)

        if weekday:
            cols.append('Weekday')
        cols.append('Date')
        imputed_df = pd.DataFrame(flat_missing_days, columns=cols)

        return imputed_df

    def remove_outliers(self, data):
        """ Detecting and removing outliers by using the standard deviation method. Threshold is set to 3. """

        #data = self.missing_days(ids, date, ordinal_col, nominal_col, groupby, sec_groupby)
        data = self.to_df(data)

        col_names = list(data.columns)
        col_names.remove('ID')
        col_names.remove('Date')

        outliers_df = data[(np.abs(stats.zscore((data[col_names])) < 3).all(axis=1))]

        return outliers_df

    def get_processed_data(self, data, ids, date, ordinal_col, nominal_col, groupby,
                           compute_usage=True, aggregate_variables=True,
                           impute_days=True, remove_outliers=True, weekday=True):

        if compute_usage:
            df_1 = self.compute_usage(data, ids, date)
        else:
            df_1 = data

        df_2 = self.treat_cat_vars(df_1, ordinal_col, nominal_col)

        if aggregate_variables:
            df_3 = self.aggregate_variables(df_2, groupby)
        else:
            df_3 = df_2

        if impute_days:
            if weekday:
                df_4 = self.missing_days(df_3, weekday=True)
            else:
                df_4 = self.missing_days(df_3, weekday=False)
        else:
            df_4 = df_3

        if remove_outliers:
            df_5 = self.remove_outliers(df_4)
        else:
            df_5 = df_4

        return df_5

    def transform_variables(self, data, ids, ordinal_col, nominal_col, cont_col,
                            outcome='Usage', transformed=True, return_scaler=True):

        #data = self.remove_outliers(ids, date, ordinal_col, nominal_col, cont_col, groupby, sec_groupby)
        data = self.to_df(data)

        ids_array = np.array(data[ids]).reshape(-1, 1)
        date_array = np.array(pd.to_datetime(data['Date'], errors='coerce').dt.strftime('%Y-%m-%d')).reshape(-1, 1)

        if transformed:
            if len(cont_col) == 1:
                scaled = self.scaler.fit_transform(np.array(data[cont_col]).reshape(-1, 1))
            else:
                scaled = self.scaler.fit_transform(np.array(data[cont_col]))

            col_names = sum([nominal_col, ordinal_col], [])
            if len(nominal_col) == 1:
                col_name = ''.join(nominal_col)
                col_names[0] = f'{col_name}_1'
                col_names.insert(1, f'{col_name}_2')

            concat = np.concatenate((np.array(data[col_names]), scaled), axis=1)
            normalised = self.normaliser.fit_transform(concat)

            # Keep a copy of only standardising and normalising the usage for easier inverse transform later.
            outcome_scaled = self.scaler.fit_transform(np.array(data[outcome]).reshape(-1, 1))
            outcome_normalised = self.normaliser.fit_transform(outcome_scaled)

            preprocessed = np.concatenate((ids_array, date_array, normalised), axis=1)
            for i in cont_col:
                col_names.append(i)
        else:
            col_names = sum([nominal_col, ordinal_col, cont_col], [])
            if len(nominal_col) == 1:
                col_name = ''.join(nominal_col)
                col_names[0] = f'{col_name}_1'
                col_names.insert(1, f'{col_name}_2')

            cols = np.array(data[col_names])
            preprocessed = np.concatenate((ids_array, date_array, cols), axis=1)

        col_names.insert(0, 'ID')
        col_names.insert(1, 'Date')
        preprocessed_df = pd.DataFrame(preprocessed, columns=col_names)

        #col_names.remove('Age')
        #col_names.insert(4, 'Age')
        preprocessed_df = preprocessed_df[col_names]

        if return_scaler:
            return self.scaler, self.normaliser
        else:
            return preprocessed_df

    def preprocess_timesteps(self, data, static_cols_lens=5, n_in=1, n_out=1, groupby=True, dropnan=True):

        data = self.to_df(data)

        # Pre-process multivariate time series, by taking the next timestamp as y (to be predicted).

        if groupby:
            agg_data = []
            for i in data['ID'].unique():
                col_index = len(data.columns)
                dropping_col_index = col_index - static_cols_lens - 1

                sub_data = data[data['ID'] == i]
                sub_data.sort_index()
                agg = utils.preprocess_multivariate_ts(sub_data.values, n_in, n_out, dropnan=dropnan)
                if n_in == 1:
                    final_agg = agg.drop(agg.iloc[:, col_index:-1], axis=1)
                elif n_in == 2:
                    agg1 = agg.drop(agg.iloc[:, col_index:col_index + static_cols_lens], axis=1)
                    final_agg = agg1.drop(agg1.iloc[:, col_index + dropping_col_index + 1:-1], axis=1)
                elif n_in == 3:
                    agg1 = agg.drop(agg.iloc[:, col_index:col_index + static_cols_lens], axis=1)
                    agg2 = agg1.drop(agg1.iloc[:, col_index + dropping_col_index + 1:
                                                  col_index + dropping_col_index + static_cols_lens + 1], axis=1)
                    final_agg = agg2.drop(agg2.iloc[:, col_index + dropping_col_index * 2 + 2:-1], axis=1)
                else:
                    raise Exception('Too many timesteps')
                agg_data.append(final_agg.values)
        else:
            col_index = len(data.columns)
            dropping_col_index = col_index - static_cols_lens - 1

            agg = utils.preprocess_multivariate_ts(data.values, n_in, n_out, dropnan=dropnan)
            if n_in == 1:
                agg_data = agg.drop(agg.iloc[:, col_index:-1], axis=1)
            elif n_in == 2:
                agg1 = agg.drop(agg.iloc[:, col_index:col_index + static_cols_lens], axis=1)
                agg_data = agg1.drop(agg1.iloc[:, col_index + dropping_col_index + 1:-1], axis=1)
            elif n_in == 3:
                agg1 = agg.drop(agg.iloc[:, col_index:col_index + static_cols_lens], axis=1)
                agg2 = agg1.drop(agg1.iloc[:, col_index + dropping_col_index + 1:
                                              col_index + dropping_col_index + static_cols_lens + 1], axis=1)
                agg_data = agg2.drop(agg2.iloc[:, col_index + dropping_col_index * 2 + 2:-1], axis=1)
            else:
                raise Exception('Too many timesteps')

        col_names = list(data.columns)

        if n_in == 1:
            col_names.remove('Date')
            col_names.insert(1, 'Date')
            column_names = col_names
        elif n_in == 2:
            column_names = utils.remove_static_cols(col_names, timesteps=2)
        elif n_in == 3:
            column_names = utils.remove_static_cols(col_names, timesteps=3)
        else:
            raise Exception('Too many timesteps')
        column_names.append('Outcome')

        if groupby:
            flat_agg_data = [x for xs in agg_data for x in xs]
        else:
            flat_agg_data = agg_data.values
        flat_agg_df = pd.DataFrame(flat_agg_data, columns=column_names)

        if not groupby:
            flat_agg_df['ID'] = flat_agg_df['ID'].astype('int64')
            flat_agg_df['Date'] = flat_agg_df['Date'].astype('datetime64[ns]')

            dtype_cols = col_names
            dtype_cols.remove('ID')
            dtype_cols.remove('Date')

            flat_agg_df[dtype_cols] = flat_agg_df[dtype_cols].astype('float64')

        return flat_agg_df

    def split_dataset_per_user(self,
                               data,
                               num_class=2,
                               regression=True,
                               return_train_data=True,
                               return_with_ids=True
                               ):

        Train, train_x, val_x, test_x, train_y, val_y, test_y = [], [], [], [], [], [], []

        # Split the data into train, validation, and test per participant.
        for i in data['ID'].unique():
            sub_data = data[data['ID'] == i]
            training_days = np.round(len(sub_data) * 0.8, 0).astype(int)

            train = sub_data.iloc[:training_days, :]
            test = sub_data.iloc[training_days:, :]

            x_train, y_train = train.iloc[:, :-1], train.iloc[:, [0, -1]]
            x_test, y_test = test.iloc[:, :-1], test.iloc[:, [0, -1]]

            validation_days = np.round(len(train) * 0.2, 0).astype(int)

            x_val = x_train.iloc[-validation_days:, :]
            y_val = y_train.iloc[-validation_days:]

            Train.append(train.values)
            train_x.append(x_train.values)
            val_x.append(x_val.values)
            test_x.append(x_test.values)
            train_y.append(y_train.values)
            val_y.append(y_val.values)
            test_y.append(y_test.values)

        train_data = [x for xs in Train for x in xs]
        train_df = pd.DataFrame(train_data, columns=data.columns)
        train_df = train_df.drop(['ID', 'Date'], axis=1)

        trainx = [x for xs in train_x for x in xs]
        valx = [x for xs in val_x for x in xs]
        testx = [x for xs in test_x for x in xs]

        # Keep a copy of the split sets with the identifiers.
        x_colnames = list(data.columns)
        x_colnames.remove('Outcome')

        trainX_df_ids = pd.DataFrame(trainx, columns=x_colnames)
        trainX_df = trainX_df_ids.drop(['ID', 'Date'], axis=1)

        valX_df_ids = pd.DataFrame(valx, columns=x_colnames)
        valX_df = valX_df_ids.drop(['ID', 'Date'], axis=1)

        testX_df_ids = pd.DataFrame(testx, columns=x_colnames)
        testX_df = testX_df_ids.drop(['ID', 'Date'], axis=1)

        trainY = [x for xs in train_y for x in xs]
        valY = [x for xs in val_y for x in xs]
        testY = [x for xs in test_y for x in xs]

        trainY_df_ids = pd.DataFrame(trainY, columns=['ID', 'Outcome'])
        trainY_df = trainY_df_ids.drop('ID', axis=1)

        valY_df_ids = pd.DataFrame(valY, columns=['ID', 'Outcome'])
        valY_df = valY_df_ids.drop('ID', axis=1)

        testY_df_ids = pd.DataFrame(testY, columns=['ID', 'Outcome'])
        testY_df = testY_df_ids.drop(['ID'], axis=1)

        trainX, trainY = np.array(trainX_df).astype('float32'), np.array(trainY_df).reshape(-1, 1).astype('float32')
        testX, testY = np.array(testX_df).astype('float32'), np.array(testY_df).reshape(-1, 1).astype('float32')
        valX, valY = np.array(valX_df).astype('float32'), np.array(valY_df).reshape(-1, 1).astype('float32')

        if not regression:
            trainY = keras.utils.to_categorical(trainY, num_classes=num_class)
            valY = keras.utils.to_categorical(valY, num_classes=num_class)
            testY = keras.utils.to_categorical(testY, num_classes=num_class)

            cla_trainY = pd.DataFrame(trainY)
            trainY_df_ids = pd.concat([trainY_df_ids, cla_trainY], axis=1)
            trainY_df_ids = trainY_df_ids.drop('Outcome', axis=1)

            cla_valY = pd.DataFrame(valY)
            valY_df_ids = pd.concat([valY_df_ids, cla_valY], axis=1)
            valY_df_ids = valY_df_ids.drop('Outcome', axis=1)

            cla_testY = pd.DataFrame(testY)
            testY_df_ids = pd.concat([testY_df_ids, cla_testY], axis=1)
            testY_df_ids = testY_df_ids.drop('Outcome', axis=1)

        trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
        valX = valX.reshape((valX.shape[0], valX.shape[1], 1))
        testX = testX.reshape((testX.shape[0], testX.shape[1], 1))

        if return_train_data:
            return train_df
        elif return_with_ids:
            return trainX_df_ids, valX_df_ids, testX_df_ids, trainY_df_ids, valY_df_ids, testY_df_ids
        else:
            return trainX, valX, testX, trainY, valY, testY

    def split_dataset(self,
                      data,
                      num_class=2,
                      regression=True,
                      return_train_data=True,
                      return_with_ids=True
                      ):

        # Split the data into train, validation, and test sets.

        training_days = np.round(len(data) * 0.8, 0).astype(int)

        train = data.iloc[:training_days, :]
        test = data.iloc[training_days:, :]

        x_train, y_train = train.iloc[:, :-1], train.iloc[:, [0, -1]]
        x_test, y_test = test.iloc[:, :-1], test.iloc[:, [0, -1]]

        validation_days = np.round(len(train) * 0.2, 0).astype(int)

        x_val = x_train.iloc[-validation_days:, :]
        y_val = y_train.iloc[-validation_days:]

        train_df = pd.DataFrame(train, columns=data.columns)
        train_df = train_df.drop(['ID', 'Date'], axis=1)

        # Keep a copy of the split sets with the identifiers.
        x_colnames = list(data.columns)
        x_colnames.remove('Outcome')

        trainX_df_ids = pd.DataFrame(x_train, columns=x_colnames)
        trainX_df = trainX_df_ids.drop(['ID', 'Date'], axis=1)

        valX_df_ids = pd.DataFrame(x_val, columns=x_colnames)
        valX_df = valX_df_ids.drop(['ID', 'Date'], axis=1)

        testX_df_ids = pd.DataFrame(x_test, columns=x_colnames)
        testX_df = testX_df_ids.drop(['ID', 'Date'], axis=1)

        trainY_df_ids = pd.DataFrame(y_train, columns=['ID', 'Outcome'])
        trainY_df = trainY_df_ids.drop('ID', axis=1)

        valY_df_ids = pd.DataFrame(y_val, columns=['ID', 'Outcome'])
        valY_df = valY_df_ids.drop('ID', axis=1)

        testY_df_ids = pd.DataFrame(y_test, columns=['ID', 'Outcome'])
        testY_df = testY_df_ids.drop(['ID'], axis=1)

        trainX = np.array(trainX_df).astype('float32')
        testX = np.array(testX_df).astype('float32')
        valX = np.array(valX_df).astype('float32')

        trainY = np.array(trainY_df).reshape(-1, 1).astype('float32')
        valY = np.array(valY_df).reshape(-1, 1).astype('float32')
        testY = np.array(testY_df).reshape(-1, 1).astype('float32')

        if not regression:
            trainY = keras.utils.to_categorical(trainY_df, num_classes=num_class)
            testY = keras.utils.to_categorical(testY_df, num_classes=num_class)
            valY = keras.utils.to_categorical(valY_df, num_classes=num_class)

        trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], 1))
        valX = valX.reshape((valX.shape[0], valX.shape[1], 1))
        testX = testX.reshape((testX.shape[0], testX.shape[1], 1))

        if return_train_data:
            return train_df
        elif return_with_ids:
            return trainX_df_ids, valX_df_ids, testX_df_ids, trainY_df_ids, valY_df_ids, testY_df_ids
        else:
            return trainX, valX, testX, trainY, valY, testY

