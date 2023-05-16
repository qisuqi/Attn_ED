import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import lime.lime_tabular
import shap
import dice_ml
from dice_ml import Dice
from AttnED_TensorFlow import utils, tuning
import time
import warnings
from itertools import cycle

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
                 num_class=2,
                 personalised_prediction=True,
                 regression=True,
                 hyper_tune=True,
                 simple_model=True,
                 prediction=None,
                 evaluation=None,
                 history=None,
                 best_hps=None):

        self.title = title
        self.epochs = epochs
        self.regression = regression
        self.hyper_tune = hyper_tune
        self.simple_model = simple_model
        self.directory = directory
        self.ids_to_predict = ids_to_predict
        self.personalised_prediction = personalised_prediction
        self.num_class = num_class

        self.prediction = prediction
        self.evaluation = evaluation
        self.history = history
        self.best_hps = best_hps

    def simple_model_builder(self, x_train, y_train, x_val, y_val, imbalanced=None):
        if self.hyper_tune:
            start_hp = time.time()

            hyper_model = tuning.SimpleHyperModel(x_train, self.regression)

            tuner = kt.Hyperband(hyper_model,
                                 objective='val_loss',
                                 max_epochs=10,
                                 factor=3,
                                 directory=self.directory,
                                 project_name=f"{self.title}")

            if self.regression:
                stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
            else:
                stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)

            tuner.search(x_train, y_train,
                         validation_data=(x_val, y_val),
                         epochs=self.epochs,
                         callbacks=[stop_early],
                         verbose=0,
                         shuffle=True)

            self.best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

            end_hp = time.time()

            print(f"""
                  The hyperparameter search is complete. 
                  The optimal number of units in the dense layer is {self.best_hps.get('unit')}, 
                  the optimal learning rate for the optimizer is {self.best_hps.get('lr')}, 
                  the optimal activation function is  {self.best_hps.get('activation')},
                  the optimal batch size is {self.best_hps.get('bs')}, 
                  the optimal drop out rate is {self.best_hps.get('dr')}.
                  """)

            print('Tuning Time: ', end_hp - start_hp)

            model = tuner.hypermodel.build(self.best_hps)

        else:
            inputs = tf.keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))

            dense1 = tf.keras.layers.Dense(64, activation='relu')(inputs)
            dense2 = tf.keras.layers.Dense(32, activation='relu')(dense1)
            dense3 = tf.keras.layers.Dense(16, activation='relu')(dense2)
            drop1 = tf.keras.layers.Dropout(0.5)(dense3)
            flatten = tf.keras.layers.Flatten()(drop1)

            if self.regression:
                if imbalanced is not None:
                    outputs = tf.keras.layers.Dense(1,
                                                    activation='tanh',
                                                    bias_initializer=imbalanced[1])(flatten)
                else:
                    outputs = tf.keras.layers.Dense(1,
                                                    activation='tanh')(flatten)

                model = tf.keras.Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                              loss=tf.keras.losses.MeanSquaredError(),
                              metrics=REGRESSION_METRICS)
            else:
                if imbalanced is not None:
                    outputs = tf.keras.layers.Dense(self.num_class,
                                                    activation='softmax',
                                                    bias_initializer=imbalanced[1])(flatten)
                else:
                    outputs = tf.keras.layers.Dense(self.num_class, activation='softmax')(flatten)

                model = tf.keras.Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                              loss=tf.keras.losses.CategoricalCrossentropy(),
                              metrics=CLASSIFICATION_METRICS)

        return model

    def model_builder(self, x_train, y_train, x_val, y_val):

        if self.hyper_tune:

            start_hp = time.time()

            hyper_model = tuning.HyperModel(x_train, self.regression, self.num_class)

            if self.regression:
                tuner = kt.Hyperband(hyper_model,
                                     objective='val_loss',
                                     max_epochs=10,
                                     factor=3,
                                     directory=self.directory,
                                     project_name=f"{self.title}")

                stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              patience=5,
                                                              mode='min')

            else:
                tuner = kt.Hyperband(hyper_model,
                                     objective='val_accuracy',
                                     max_epochs=10,
                                     factor=3,
                                     directory=self.directory,
                                     project_name=f"{self.title}")

                stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                              patience=5,
                                                              mode='max')
            print('Start searching...')
            tuner.search(x_train, y_train,
                         validation_data=(x_val, y_val),
                         epochs=self.epochs,
                         callbacks=[stop_early],
                         verbose=0,
                         shuffle=True)

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
                  the optimal drop out rate for the LSTM layer is {self.best_hps.get('drop_lstm')}, and
                  the optimal recurrent drop out rate for the LSTM layer is {self.best_hps.get('drop_lstm_rec')}.
                  """)

            model = tuner.hypermodel.build(self.best_hps)

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
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                              loss=tf.keras.losses.MeanSquaredError(),
                              metrics=REGRESSION_METRICS)
            else:
                outputs = tf.keras.layers.Dense(self.num_class, activation='softmax')(dense)

                model = tf.keras.Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer=tf.keras.optimizers.Adam(),
                              loss=tf.keras.losses.CategoricalCrossentropy(),
                              metrics=CLASSIFICATION_METRICS)

        return model

    def train_model(self, x_train, x_val, y_train, y_val, name, verbose, imbalanced=None):

        if self.simple_model:
            model = self.simple_model_builder(x_train, y_train, x_val, y_val, imbalanced)
        else:
            model = self.model_builder(x_train, y_train, x_val, y_val)

        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        if self.regression:
            es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.epochs / 10, mode='min')
        else:
            es = tf.keras.callbacks.EarlyStopping(monitor='val_prc', patience=self.epochs / 10, mode='max')

        start = time.time()

        if self.hyper_tune:
            if imbalanced is not None:
                class_weight = imbalanced[0]
                self.history = model.fit(x_train,
                                         y_train,
                                         validation_data=(x_val, y_val),
                                         epochs=self.epochs,
                                         batch_size=self.best_hps.get('bs'),
                                         verbose=verbose,
                                         callbacks=[es],
                                         shuffle=True,
                                         class_weight=class_weight)
            else:
                self.history = model.fit(x_train,
                                         y_train,
                                         validation_data=(x_val, y_val),
                                         epochs=self.epochs,
                                         batch_size=self.best_hps.get('bs'),
                                         verbose=verbose,
                                         callbacks=[es],
                                         shuffle=True)
        else:
            if imbalanced is not None:
                class_weight = imbalanced[0]
                self.history = model.fit(x_train,
                                         y_train,
                                         validation_data=(x_val, y_val),
                                         epochs=self.epochs,
                                         batch_size=2048,
                                         verbose=verbose,
                                         callbacks=[es],
                                         class_weight=class_weight)
            else:
                self.history = model.fit(x_train,
                                         y_train,
                                         validation_data=(x_val, y_val),
                                         epochs=self.epochs,
                                         batch_size=2048,
                                         verbose=verbose,
                                         callbacks=[es])
        end = time.time()
        print('Training Time: ', end - start)

        path = f"saved_models/{self.title}_bestmodel_{name}.h5"
        model.save(path)
        return path

    def predict(self, x_test_ids, y_test_ids, scaler, normaliser, saved_model_path):

        model = tf.keras.models.load_model(saved_model_path)

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

        if self.regression:
            inv_test = normaliser.inverse_transform(prediction)
            inv_test = scaler.inverse_transform(inv_test)
            self.prediction = inv_test
            self.evaluation = evaluation
        else:
            self.prediction = prediction
            self.evaluation = evaluation

        return self.prediction, self.evaluation

    def get_attention(self,
                      saved_model_path,
                      x_test_ids,
                      instance_to_explain,
                      return_attention_types: str):

        model = tf.keras.models.load_model(saved_model_path)

        attention_model = tf.keras.Model(inputs=model.input, outputs=[model.output, model.layers[2].output])

        if self.personalised_prediction:
            test_x = x_test_ids[x_test_ids['ID'] == self.ids_to_predict]
            test_x = test_x.drop(['ID', 'Date'], axis=1)
            test_x_array = np.array(test_x)
        else:
            test_x = x_test_ids.drop(['ID', 'Date'], axis=1)
            test_x_array = np.array(test_x)

        prediction, attn_weights = attention_model.predict(test_x_array)

        # The result of the computation
        attention_output = np.asarray(attn_weights[0])
        individual_attention = attention_output[instance_to_explain]

        # Multi-head attention coefficients over attention axes
        attention_score = np.asarray(attn_weights[1])

        if return_attention_types == 'one instance all timesteps':

            return prediction, individual_attention

        elif return_attention_types == 'one instance mean timestep':
            attention_oimt = np.average(individual_attention, axis=1).reshape(-1, 1)

            return prediction, attention_oimt

        elif return_attention_types == 'all instances all timesteps':

            return prediction, attention_output

        elif return_attention_types == 'all instances mean timestep':
            attention_aimt = np.average(attention_output, axis=2)

            return prediction, attention_aimt

        elif return_attention_types == 'scores':

            # attention score of shape (no.instances, no.features, no.features)
            attention_score = attention_score.reshape((attention_score.shape[0],
                                                      attention_score.shape[2],
                                                      attention_score.shape[3]))

            return prediction, attention_score

        else:
            raise Exception('...')

    def plot_attention(self,
                       attention,
                       org_xtest,
                       scaler,
                       normaliser,
                       feature_names,
                       instance_to_explain=0):

        prediction, attn_weights = attention

        if not self.regression:
            prediction = np.argmax(prediction, axis=1).reshape(-1, 1)

        inv_prediction = normaliser.inverse_transform(prediction)
        inv_prediction = scaler.inverse_transform(inv_prediction)

        x_test_to_explain = org_xtest[instance_to_explain, :]

        y_labels = []

        for i in range(len(feature_names)):
            y_label = f'{feature_names[i]}={x_test_to_explain[i]}'
            y_labels.append(y_label)

        y_labels = np.array(y_labels)

        ax = plt.figure(figsize=(10, 5)).gca()
        img = ax.matshow(attn_weights)

        ax.set_yticks(range(len(x_test_to_explain)))
        ax.set_yticklabels(y_labels)

        if attn_weights.shape[1] == 1:
            ax.set_xticks(range(len(inv_prediction[instance_to_explain])), labels=inv_prediction[instance_to_explain])
            plt.colorbar(img, ax=ax)
        else:
            plt.colorbar(img, ax=ax, orientation='horizontal')

        plt.show()

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
                     dice_verbose=True):

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
            train_x_array = np.array(train_x).astype('float32')

            org_train_x = org_x_train_df_ids.drop(['ID', 'Date'], axis=1)
            org_train_x_array = np.array(org_train_x).astype('float32')

            train_y = y_train_df_ids.drop(['ID'], axis=1)
            train_y_array = np.array(train_y).astype('float32')

            test_x = x_test_df_ids.drop(['ID', 'Date'], axis=1)
            test_x_array = np.array(test_x).astype('float32')

            org_test_x = org_x_test_df_ids.drop(['ID', 'Date'], axis=1)
            org_test_x_array = np.array(org_test_x).astype('float32')

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

            shap_values = shap_explainer.shap_values(test_x_array)

            shap.initjs()
            shap.summary_plot(shap_values=shap_values,
                              features=test_x_array,
                              feature_names=feature_names)
            shap.summary_plot(shap_values=shap_values[-1],
                              features=test_x_array,
                              feature_names=feature_names)

            return shap_values

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

            if self.regression:
                m = dice_ml.Model(model_path=saved_model_path,
                                  backend=backend,
                                  model_type='regressor')
            else:
                m = dice_ml.Model(model_path=saved_model_path,
                                  backend=backend,
                                  model_type='classifier')
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

        if self.regression:
            prediction = self.prediction

            ids_y = y_test_ids[y_test_ids['ID'] == self.ids_to_predict]
            ids_y = ids_y.drop(['ID'], axis=1)
            true = np.array(ids_y).reshape(-1, 1)
        else:
            prediction = np.argmax(self.prediction, axis=1)

            y_test_ids['args'] = np.argmax(np.array(y_test_ids.iloc[:, 1:]), axis=1)

            ids_y = y_test_ids[y_test_ids['ID'] == self.ids_to_predict]
            ids_y = ids_y.drop(ids_y.iloc[:, :-1], axis=1)
            true = np.array(ids_y).reshape(-1, 1)

        utils.plot_results(date, true, prediction)

    def get_error_estimators(self, y_test_ids, prediction, train):
        """Calculate the error estimators. """

        if self.personalised_prediction:
            ids_y = y_test_ids[y_test_ids['ID'] == self.ids_to_predict]
            ids_y = ids_y.drop(['ID'], axis=1)
            ids_y = np.array(ids_y).reshape(1, -1)

            prediction = prediction.reshape(1, -1)

            train = np.array(train)

            utils.error_estimator(ids_y, prediction, train)

        else:
            true = np.array(y_test_ids['Outcome']).reshape(-1, 1)
            utils.error_estimator(true, prediction, train)

    def get_cm(self, y_test_ids):

        y_test = y_test_ids.copy()

        y_test['args'] = np.argmax(np.array(y_test.iloc[:, 1:]), axis=1)
        prediction = np.argmax(self.prediction, axis=1).reshape(-1, 1)

        if self.personalised_prediction:
            ids_y = y_test[y_test['ID'] == self.ids_to_predict]
        else:
            ids_y = y_test
        ids_y = ids_y.drop(ids_y.iloc[:, :-1], axis=1)
        true = np.array(ids_y)

        xlabel = list(np.unique(prediction))
        ylabel = list(np.unique(true))

        if set(ylabel) - set(xlabel) is not None:
            x_append = list(set(ylabel) - set(xlabel))
            y_append = list(set(xlabel) - set(ylabel))

            xlabel.extend(x_append)
            ylabel.extend(y_append)

        xlabel.sort()
        ylabel.sort()

        cm = metrics.confusion_matrix(true, prediction)

        return cm, ylabel, xlabel

    def plot_cm(self, y_test_ids):

        cm, ylabel, xlabel = self.get_cm(y_test_ids)

        sns.heatmap(cm,
                    annot=True,
                    yticklabels=ylabel,
                    xticklabels=xlabel)
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        plt.show()

    def roc_auc_score(self, y_test_ids):

        y_test = y_test_ids.copy()

        y_test['args'] = np.argmax(np.array(y_test.iloc[:, 1:]), axis=1)
        prediction = np.argmax(self.prediction, axis=1).reshape(-1, 1)

        if self.personalised_prediction:
            ids_y = y_test[y_test['ID'] == self.ids_to_predict]
        else:
            ids_y = y_test
        ids_y = ids_y.drop(ids_y.iloc[:, :-1], axis=1)

        auc = utils.roc_auc_score_multiclass(ids_y['args'], prediction, return_type='auc')
        return auc

    def plot_roc(self, y_test_ids):

        if self.personalised_prediction:
            ids_y = y_test_ids[y_test_ids['ID'] == self.ids_to_predict]
        else:
            ids_y = y_test_ids
        ids_y = ids_y.drop('ID', axis=1)

        arg_label = np.argmax(np.array(ids_y), axis=1)
        num_class = np.unique(arg_label)

        fig, ax = plt.subplots(figsize=(6, 6))

        colors = cycle(["aqua", "darkorange", "cornflowerblue"])

        for class_id, color in zip(num_class, colors):
            metrics.RocCurveDisplay.from_predictions(
                np.array(ids_y)[:, class_id],
                self.prediction[:, class_id],
                name=f"ROC curve for {class_id}",
                color=color,
                ax=ax,
            )

        plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.show()

        plt.show()
