import tensorflow as tf
import keras_tuner as kt

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


class HyperModel(kt.HyperModel):

    def __init__(self, x_train, regression):
        self.trainX = x_train
        self.regression = regression

    def build(self, hp):
        """Define the model architecture for hyper-tuning."""

        lstm_unit = hp.Choice('lstm_unit', values=[32, 64, 128, 216, 512])
        lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        dr = hp.Float("dr", min_value=1e-4, max_value=1e-2, sampling="log")
        drop_blstm = hp.Float("drop_blstm", min_value=1e-4, max_value=1e-2, sampling="log")
        drop_blstm_rec = hp.Float("drop_blstm_rec", min_value=1e-4, max_value=1e-2, sampling="log")
        activation = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh'])

        inputs = tf.keras.layers.Input(shape=(self.trainX.shape[1], self.trainX.shape[2]))
        lstm = tf.keras.layers.LSTM(lstm_unit,
                                    return_sequences=True,
                                    activation='tanh',
                                    recurrent_activation='sigmoid',
                                    dropout=drop_blstm,
                                    recurrent_dropout=drop_blstm_rec,
                                    kernel_initializer='glorot_normal',
                                    recurrent_initializer='he_uniform')(inputs)

        attention = tf.keras.layers.MultiHeadAttention(num_heads=1, key_dim=3)
        attn_out, attention_weights = attention(lstm, lstm, return_attention_scores=True)

        drop = tf.keras.layers.Dropout(dr)(attn_out)

        lstm1 = tf.keras.layers.LSTM(lstm_unit,
                                     return_sequences=False,
                                     activation='tanh',
                                     recurrent_activation='sigmoid',
                                     dropout=drop_blstm,
                                     recurrent_dropout=drop_blstm_rec,
                                     kernel_initializer='glorot_normal',
                                     recurrent_initializer='he_uniform')(drop)

        dense = tf.keras.layers.Dense(lstm_unit, activation='relu')(lstm1)

        if self.regression:
            outputs = tf.keras.layers.Dense(1, activation=activation)(dense)

            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss=tf.keras.losses.MeanSquaredError(),
                          metrics=REGRESSION_METRICS)
        else:
            outputs = tf.keras.layers.Dense(2, activation='softmax')(dense)

            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                          loss=tf.keras.losses.CategoricalCrossentropy(),
                          metrics=CLASSIFICATION_METRICS)
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args, batch_size=hp.Choice("bs", [16, 32, 64]), **kwargs)
