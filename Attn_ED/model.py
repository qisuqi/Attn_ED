import numpy as np
import tensorflow as tf
import keras_tuner as kt
from numpy import ndarray
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import lime.lime_tabular
import shap
import dice_ml
from dice_ml import Dice
from Attn_ED import utils, tuning
import time
import warnings

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Pre define metrics for training for classification problems
CLASSIFICATION_METRICS = [tf.keras.metrics.TruePositives(name='tp'),
                          tf.keras.metrics.FalsePositives(name='fp'),
                          tf.keras.metrics.TrueNegatives(name='tn'),
                          tf.keras.metrics.FalseNegatives(name='fn'),
                          tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                          tf.keras.metrics.Precision(name='precision'),
                          tf.keras.metrics.Recall(name='recall'),
                          tf.keras.metrics.AUC(name='auc')]

# Pre define metrics for training for regression problems
REGRESSION_METRICS = [tf.keras.metrics.MeanSquaredError(name="mse"),
                      tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                      tf.keras.metrics.MeanAbsoluteError(name="mae"),
                      tf.keras.metrics.MeanAbsolutePercentageError(name="mape")]


class AttnED:
    def __init__(self,
                 title,
                 epochs,
                 directory,
                 ids_to_predict=0,
                 personalised_prediction=True,
                 regression=True,
                 hyper_tune=True,
                 prediction=None,
                 evaluation=None,
                 history=None,
                 best_hps=None):

        self.title = title
        self.epochs = epochs
        self.regression = regression
        self.directory = directory
        self.hyper_tune = hyper_tune
        self.ids_to_predict = ids_to_predict
        self.personalised_prediction = personalised_prediction

        self.prediction = prediction
        self.evaluation = evaluation
        self.history = history
        self.best_hps = best_hps

    def model_builder(self, x_train, y_train, x_val, y_val):
        if self.hyper_tune:
            start_hp = time.time()
            hyper_model = tuning.HyperModel(x_train, regression=self.regression)

            tuner = kt.Hyperband(hyper_model,
                                 objective='val_loss',
                                 max_epochs=10,
                                 factor=3,
                                 directory=self.directory,
                                 project_name=f"{self.title}")

            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

            tuner.search(x_train, y_train, validation_data=(x_val, y_val), epochs=self.epochs, callbacks=[stop_early],
                         verbose=0, shuffle=True)
            self.best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

            end_hp = time.time()
            print('Tuning Time: ', end_hp - start_hp)

            print(f"""
                   The hyperparameter search is complete. 
                   The optimal number of units in the lstm layer is {self.best_hps.get('lstm_unit')}, 
                   the optimal learning rate for the optimizer is {self.best_hps.get('lr')}, 
                   the optimal activation function is  {self.best_hps.get('activation')},
                   the optimal batch size is {self.best_hps.get('bs')}, 
                   the optimal drop out rate is {self.best_hps.get('dr')},
                   the optimal drop out rate for the BLSTM layer is {self.best_hps.get('drop_blstm')}, and
                   the optimal recurrent drop out rate for the BLSTM layer is {self.best_hps.get('drop_blstm_rec')}.
                   """)

            model = tuner.hypermodel.build(self.best_hps)

            return model
        else:
            inputs = tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))

            lstm = tf.keras.layers.LSTM(64,
                                        return_sequences=True,
                                        activation='tanh',
                                        recurrent_activation='sigmoid',
                                        dropout=0.000287531,
                                        recurrent_dropout=0.00084181157,
                                        kernel_initializer='glorot_normal',
                                        recurrent_initializer='he_uniform')(inputs)

            attention = tf.keras.layers.MultiHeadAttention(num_heads=1, key_dim=3)
            attn_out, attention_weights = attention(lstm, lstm, return_attention_scores=True)

            drop = tf.keras.layers.Dropout(0.00940133)(attn_out)

            lstm1 = tf.keras.layers.LSTM(64,
                                         return_sequences=False,
                                         activation='tanh',
                                         recurrent_activation='sigmoid',
                                         dropout=0.000287531,
                                         recurrent_dropout=0.00084181157,
                                         kernel_initializer='glorot_normal',
                                         recurrent_initializer='he_uniform')(drop)

            dense = tf.keras.layers.Dense(64, activation='relu')(lstm1)

            if self.regression:
                outputs = tf.keras.layers.Dense(1, activation='tanh')(dense)

                model = tf.keras.Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0004217),
                              loss=tf.keras.losses.MeanSquaredError(),
                              metrics=REGRESSION_METRICS)
            else:
                outputs = tf.keras.layers.Dense(2, activation='softmax')(dense)

                model = tf.keras.Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer=tf.keras.optimizers.Adam(),
                              loss=tf.keras.losses.CategoricalCrossentropy(),
                              metrics=CLASSIFICATION_METRICS)

            return model

    def train_model(self, x_train, x_val, y_train, y_val, name, verbose):

        model = self.model_builder(x_train, y_train, x_val, y_val)

        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.epochs / 5)

        start = time.time()

        if self.hyper_tune:
            self.history = model.fit(x_train,
                                     y_train,
                                     validation_data=(x_val, y_val),
                                     epochs=self.epochs,
                                     batch_size=self.best_hps.get('bs'),
                                     verbose=verbose,
                                     callbacks=[es],
                                     shuffle=True)
        else:
            self.history = model.fit(x_train,
                                     y_train,
                                     validation_data=(x_val, y_val),
                                     epochs=self.epochs,
                                     batch_size=16,
                                     verbose=verbose,
                                     callbacks=[es])
        end = time.time()
        print('Training Time: ', end - start)

        path = f"saved_models/{self.title}_bestmodel_{name}.h5"
        model.save(path)
        return path

    #def predict(self, x_train, x_val, x_test, y_train, y_val, y_test, x_test_ids, y_test_ids,
     #           scaler, normaliser, saved_model_path):
    def predict(self, x_test, y_test, x_test_ids, y_test_ids, scaler, normaliser, saved_model_path):
        #if saved_model_path:
        model = tf.keras.models.load_model(saved_model_path)
        #else:
            #path = self.train_model(x_train, x_val, y_train, y_val, 'v1')
            #model = tf.keras.models.load_model(path)

        if self.regression:
            if self.personalised_prediction:
                testX_ids = x_test_ids[x_test_ids['ID'] == self.ids_to_predict]
                testX_ids = testX_ids.drop(['ID', 'Date'], axis=1)

                testY_ids = y_test_ids[y_test_ids['ID'] == self.ids_to_predict]
                testY_ids = testY_ids.drop(['ID'], axis=1)

                prediction = model.predict(testX_ids)
                evaluation = model.evaluate(testX_ids, testY_ids, verbose=0)
            else:
                testX_ids = x_test_ids.drop(['ID', 'Date'], axis=1)
                testY_ids = y_test_ids.drop(['ID'], axis=1)

                prediction = model.predict(testX_ids)
                evaluation = model.evaluate(testX_ids, testY_ids)

            inv_test = normaliser.inverse_transform(prediction)
            inv_test = scaler.inverse_transform(inv_test)
            self.prediction = inv_test
            self.evaluation = evaluation
        else:
            self.prediction = model.predict(x_test)
            self.evaluation = model.evaluate(x_test, y_test)

        return self.prediction, self.evaluation

    def plot_attention(self, saved_model_path, x_test, org_xtest, scaler, normaliser, instance_to_explain, feature_names):

        model = tf.keras.models.load_model(saved_model_path)

        attention_model = tf.keras.Model(inputs=model.input, outputs=[model.output, model.layers[2].output])
        prediction, attn_weights = attention_model.predict(x_test)

        attention = np.asarray(attn_weights[1])[instance_to_explain]
        reshape_attention = attention.reshape(attention.shape[1], attention.shape[2])
        avg_attention = np.average(reshape_attention, axis=1).reshape(attention.shape[2], 1)

        inv_prediction = normaliser.inverse_transform(prediction)
        inv_prediction = scaler.inverse_transform(inv_prediction)

        x_test_to_explain = org_xtest[instance_to_explain, :]

        y_labels = []

        for i in range(len(feature_names)):
            y_label = f'{feature_names[i]}={x_test_to_explain[i]}'
            y_labels.append(y_label)

        y_labels = np.array(y_labels)

        plt.figure(figsize=(40, 10))
        ax = plt.gca()
        img = ax.matshow(avg_attention)

        ax.set_xticks(range(len(inv_prediction[instance_to_explain])), labels=inv_prediction[instance_to_explain])
        ax.set_yticks(range(len(x_test_to_explain)))

        ax.set_yticklabels(y_labels)

        plt.colorbar(img, ax=ax)

        plt.show()

        return

    def explanations(self,
                     orig_train,
                     org_x_train_df_ids,
                     org_x_test_df_ids,
                     x_train_df_ids,
                     x_test_df_ids,
                     y_train_df_ids,
                     scaler,
                     normaliser,
                     desired_range,
                     feature_names,
                     local_instance,
                     saved_model_path,
                     method='shap',
                     dice_verbose=True,
                     return_shap_values=True):

        model = tf.keras.models.load_model(saved_model_path)

        if self.personalised_prediction:
            train_x = x_train_df_ids[x_train_df_ids['ID'] == self.ids_to_predict]
            train_x = train_x.drop(['ID', 'Date'], axis=1)
            train_x_array = np.array(train_x)

            org_train_x = org_x_train_df_ids[org_x_train_df_ids['ID'] == self.ids_to_predict]
            org_train_x = org_train_x.drop(['ID', 'Date'], axis=1)
            org_train_x_array = np.array(org_train_x)

            train_y = y_train_df_ids[y_train_df_ids['ID'] == self.ids_to_predict]
            train_y = train_y.drop(['ID'], axis=1)
            train_y_array = np.array(train_y)

            test_x = x_test_df_ids[x_test_df_ids['ID'] == self.ids_to_predict]
            test_x = test_x.drop(['ID', 'Date'], axis=1)
            test_x_array = np.array(test_x)

            org_test_x = org_x_test_df_ids[org_x_test_df_ids['ID'] == self.ids_to_predict]
            org_test_x = org_test_x.drop(['ID', 'Date'], axis=1)
            org_test_x_array = np.array(org_test_x)
        else:
            train_x = x_train_df_ids.drop(['ID', 'Date'], axis=1)
            train_x_array = np.array(train_x)

            org_train_x = org_x_train_df_ids.drop(['ID', 'Date'], axis=1)
            org_train_x_array = np.array(org_train_x)

            train_y = y_train_df_ids.drop(['ID'], axis=1)
            train_y_array = np.array(train_y)

            test_x = x_test_df_ids.drop(['ID', 'Date'], axis=1)
            test_x_array = np.array(test_x)

            org_test_x = org_x_test_df_ids.drop(['ID', 'Date'], axis=1)
            org_test_x_array = np.array(org_test_x)

        if method == 'lime':

            lime_train_x = np.reshape(org_train_x_array, (org_train_x_array.shape[0], org_train_x_array.shape[1], 1))
            lime_test_x = np.reshape(org_test_x_array, (org_test_x_array.shape[0], org_test_x_array.shape[1], 1))

            lime_explainer = lime.lime_tabular.RecurrentTabularExplainer(lime_train_x,
                                                                         training_labels=train_y_array,
                                                                         feature_names=feature_names,
                                                                         mode='regression')
            assert local_instance <= lime_test_x.shape[0]

            def prediction(lime_test_x):
                pred = model.predict(lime_test_x)
                return utils.inverse_transform(scaler, normaliser, pred, dim=2)

            exp = lime_explainer.explain_instance(lime_test_x[local_instance], prediction)
            exp.show_in_notebook(show_table=True)

        elif method == 'shap':
            warnings.filterwarnings("ignore")

            shap_explainer = shap.KernelExplainer(model=model, data=train_x_array[:50], link='identity')

            shap_values = shap_explainer.shap_values(test_x)
            train_shap_values = shap_explainer.shap_values(train_x)

            shap.initjs()
            shap.summary_plot(shap_values=shap_values,
                              features=test_x_array,
                              feature_names=feature_names)
            shap.summary_plot(shap_values=shap_values[-1],
                              features=test_x_array,
                              feature_names=feature_names)

            if return_shap_values:
                return shap_values, train_shap_values

        elif method == 'dice':

            backend = 'TF' + tf.__version__[0]

            features_dict = {}
            for i in range(len(feature_names)):
                dicts = {feature_names[i]: [int(min(orig_train[feature_names[i]])),
                                            int(np.ceil(max(orig_train[feature_names[i]])))]}

                features_dict.update(dicts)

            d = dice_ml.Data(features=features_dict,
                             outcome_name='Outcome',
                             outcome_range=[int(min(orig_train['Outcome'])),
                                            int(max(orig_train['Outcome']))])

            m = dice_ml.Model(model_path=saved_model_path, backend=backend, model_type='regressor')

            exp = Dice(d, m)

            data_to_explain = org_test_x_array[local_instance].reshape(-1, 1)#.astype(int)
            data_dicts = dict(enumerate(data_to_explain.flatten(), 1))
            query_instances = dict(zip(feature_names, list(data_dicts.values())))
            genetic = exp.generate_counterfactuals(query_instances,
                                                   desired_range=desired_range,
                                                   total_CFs=5,
                                                   verbose=dice_verbose)

            genetic.visualize_as_dataframe(show_only_changes=True)

        else:
            raise Exception("This XAI method is not currently supported.")

    def plot_loss_acc_classification(self):
        """Plot loss and accuracy curves."""

        utils.plot_acc(self.history.history['accuracy'],
                       self.history.history['val_accuracy'])

        utils.plot_loss(self.history.history['loss'],
                        self.history.history['val_loss'])

    def plot_loss_regression(self):
        """Plot loss curves"""

        utils.plot_loss(self.history.history['loss'],
                        self.history.history['val_loss'])

        utils.plot_rmse(self.history.history['rmse'],
                        self.history.history['val_rmse'])

    def plot_prediction(self, x_test_ids, y_test_ids):
        """Plot predicted time series against true time series"""
        # TODO: Need to change the dates to the outcome dates

        trainX = x_test_ids[x_test_ids['ID'] == self.ids_to_predict]

        date = trainX['Date']

        ids_y = y_test_ids[y_test_ids['ID'] == self.ids_to_predict]
        ids_y = ids_y.drop(['ID'], axis=1)
        true = np.array(ids_y).reshape(-1, 1)

        prediction = self.prediction.reshape(-1, 1)

        utils.plot_results(date, true, prediction)

    def get_error_estimators(self, y_test_ids, train, og_y_test):
        """Calculate the error estimators. """

        if self.personalised_prediction:
            ids_y = y_test_ids[y_test_ids['ID'] == self.ids_to_predict]
            ids_y = ids_y.drop(['ID'], axis=1)
            ids_y = np.array(ids_y).reshape(1, -1)

            prediction = self.prediction.reshape(1, -1)

            train = np.array(train)

            utils.error_estimator(ids_y, prediction, train)

        else:
            utils.error_estimator(og_y_test, self.prediction, train)
