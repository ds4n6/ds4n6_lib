import os
import glob
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import losses
import matplotlib.pyplot as plt
import pandas as pd


def convert_prefetch_ham_to_hml(df):
    df_split = df['file_referenced'].str.split("\\",expand=True)
    df_split = df_split.drop(columns=[0]).fillna(value='')
    
    first_column = df_split.iloc[:, 0]
    medium_column = []
    last_column = []
    for i in range(df_split.shape[0]):
        arr = [x for x in df_split.iloc[i, 1:] if x != '']
        medium_column.append('\\'.join(arr[:-1]))
        last_column.append('\\'.join(arr[-1:])) # [-1:] because some len(arr) == 0 
    
    list_to_df = list(zip(first_column, medium_column, last_column, df['machine_id']))
    new_df = pd.DataFrame(list_to_df, columns =['A', 'B', 'C', 'machine_id'])
    return new_df


def ml_prefetch_anomalies(df, odalg="simple_autoencoder", latent_dim = 128, epochs = 10, learning_rate = 1e-3):    
    # Deep Learning
    x_train = pd.get_dummies(df).to_numpy()
    
    class Autoencoder(Model):
      def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim   
        self.encoder = layers.Dense(latent_dim, activation='relu')
        self.decoder = layers.Dense(input_dim, activation='sigmoid')

      def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def train_autoencoder(latent_dim, epochs, learning_rate):
        autoencoder = Autoencoder(input_dim=x_train.shape[1], latent_dim=latent_dim)
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        autoencoder.compile(optimizer=opt, loss=losses.MeanSquaredError())
        history = autoencoder.fit(x_train, x_train, epochs=epochs, shuffle=True, verbose=0)
        return autoencoder, history

    model, history = train_autoencoder(latent_dim=latent_dim,
                                       epochs=epochs,
                                       learning_rate=learning_rate)


    preds = model.predict(x_train)
    inference_losses = tf.keras.metrics.mean_squared_error(preds, x_train.astype('float')).numpy()
    
    ranking = []
    for i, loss in zip(range(len(inference_losses)), inference_losses):
        fr = '\\'.join(df.iloc[i, :3])
        
        machine_id = df.iloc[i]['machine_id']
        if fr.endswith('.DLL'):
            ranking.append((loss, i, fr, machine_id))

    ranking = sorted(ranking, key=lambda x: -x[0])
    anomdf = pd.DataFrame(ranking, columns=['loss', 'source_index', 'file referenced', 'machine_id'])
    return anomdf[['file referenced', 'machine_id']]
