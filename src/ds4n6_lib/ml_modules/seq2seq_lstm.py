#
# Description: implementation of ML model: seq2seq - LSTM
#

#############################################################################################
# INFO
#############################################################################################

#############################################################################################
# IMPORTS
#############################################################################################
import re, string, os, time
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf


#############################################################################################
# Class: Seq2seqData
#############################################################################################
class Seq2seqData:
    def __init__(self):
        self.batch_size = 16
        self.vocab_size = 0
        self.sequence_length = 0
        self.in_vectorization = TextVectorization()
        self.out_vectorization = TextVectorization()
        self.train_dset = 0

    def load_path_dataset(self, lm_dset, from_date, to_date, min_count):
        lm_dset['time'] = pd.to_datetime(lm_dset['time'], format='%Y-%m-%d')
        lm_dset = lm_dset[(lm_dset['time'] >= from_date) & (lm_dset['time'] <= to_date)]
        model = Word2Vec(list(lm_dset['path']), vector_size=0, min_count=min_count)
        node_list = model.wv.index_to_key
        self.vocab_size = len(node_list)+5
        
        ndset = lm_dset.copy()
        for idx,row in lm_dset.iterrows():
            for node in row.path:
                if node not in node_list:
                    ndset = ndset.drop(index=idx)
                    break
        return ndset
    
    def process_train_data(self, lm_dset):
        target_data = []
        for i in lm_dset['path']:
            target_data.append(['[sos]'] + i + ['[eos]'])
        self.sequence_length = max(len(s) for s in target_data)

        train_in  = [' '.join(i) for i in lm_dset['path']]
        train_out = [' '.join(i) for i in target_data]
        return train_in, train_out
    
    def build_train_dset(self, train_in, train_out):
        self._tokenizer(train_in, train_out) 
        t_in = self.in_vectorization(train_in)
        t_out = self.out_vectorization(train_out)
        dataset = tf.data.Dataset.from_tensor_slices((t_in, t_out[:, :-1], t_out[:, 1:]))
        self.train_dset = dataset.shuffle(len(train_in)).batch(self.batch_size, drop_remainder=True)
    
    # AUX. FUNCTIONS
    def _custom_standardization(self, input_string):
        strip_chars = string.punctuation
        strip_chars = strip_chars.replace("[", "")
        strip_chars = strip_chars.replace("]", "")
        strip_chars = strip_chars.replace("-", "")
        lowercase = tf.strings.lower(input_string)
        return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")
    
    def _tokenizer(self, train_in, train_out):
        self.in_vectorization = TextVectorization(max_tokens=self.vocab_size, output_mode="int", output_sequence_length=self.sequence_length)
        self.out_vectorization = TextVectorization(max_tokens=self.vocab_size, output_mode="int", output_sequence_length=self.sequence_length + 1, standardize=self._custom_standardization)
        self.in_vectorization.adapt(train_in)
        self.out_vectorization.adapt(train_out)


#############################################################################################
# Class: Autoencoder (LSTM)
#############################################################################################
class Autoencoder:
    def __init__(self, embed_dim, latent_dim, data):
        self.epochs = 10
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.data = data
        self.encoder = None
        self.decoder = None

    def set_epochs(self, epochs):
        self.epochs = epochs

    def build_autoencoder(self):
        self.encoder = Encoder(self.data.vocab_size, self.embed_dim, self.latent_dim)
        self.decoder = Decoder(self.data.vocab_size, self.embed_dim, self.latent_dim)
        
    def fit_autoencoder(self):
        optimizer = tf.keras.optimizers.Adam(clipnorm=5.0)
        checkpoint_dir = './training_ckpt_seq2seq'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=self.encoder, decoder=self.decoder)
        losses, accuracies = self._main_train(self.encoder, self.decoder, self.data.train_dset, self.epochs, self.data.batch_size, optimizer, checkpoint, checkpoint_prefix)
        
    def decode_sequence(self, input_sentence, node_index_dict):
        decoded_err = []
        t_path = input_sentence.split(' ') + (['[eos]']*(self.data.sequence_length))
        tokenized_input_sentence = self.data.in_vectorization([input_sentence])
        en_initial_states = self.encoder.init_states(1)
        en_outputs = self.encoder(tf.constant(tokenized_input_sentence), en_initial_states)
        de_state_h, de_state_c = en_outputs[1:]  
        
        decoded_path = '[sos]'
        for i in range(self.data.sequence_length):
            tokenized_target_sentence = self.data.out_vectorization([decoded_path])[:, :-1]
            de_output, de_state_h, de_state_c, predictions = self.decoder(tokenized_target_sentence, (de_state_h, de_state_c))
            
            n = t_path.pop(0)
            index = next((i for i, node in node_index_dict.items() if node == n), None)
            err = np.array(predictions)[0][i][index]
            decoded_err.append(err)

            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = node_index_dict[sampled_token_index]
            decoded_path += ' ' + sampled_token

            if sampled_token == '[eos]':
                break
        return decoded_path, decoded_err

    def get_anomalies(self, train_in):
        node_vocab = self.data.out_vectorization.get_vocabulary()
        node_index_dict = dict(zip(range(len(node_vocab)), node_vocab))

        e_matrix = []
        test_in_paths = [pair for pair in train_in]   
        for idx,path in enumerate(test_in_paths):
            dec_lm, err = self.decode_sequence(path, node_index_dict)
            mse = np.square(err).mean()
            e_matrix.append([idx, mse])
        error_matrix = np.array(e_matrix)
        error_matrix = error_matrix[error_matrix[:, 1].argsort()]
        return error_matrix
    
    # AUX. FUNCTIONS
    def _main_train(self, encoder, decoder, dataset, n_epochs, batch_size, optimizer, checkpoint, checkpoint_prefix):
        losses = []
        accuracies = []
        print('Model: "LSTM"')
        print('____________________________________________________________')
        for e in range(n_epochs):
            start = time.time()
            en_initial_states = encoder.init_states(batch_size)
            for batch, (input_seq, target_seq_in, target_seq_out) in enumerate(dataset.take(-1)):
                loss, accuracy = self._train_step(input_seq, target_seq_in, target_seq_out, en_initial_states, optimizer)

                if batch % 100 == 0:
                    losses.append(loss)
                    accuracies.append(accuracy)
                    print('Epoch {} Batch {} Loss {:.4f} Acc:{:.4f}'.format(e + 1, batch, loss.numpy(), accuracy.numpy()))
            if (e + 1) % 2 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
            print('Time taken for 1 epoch {:.4f} sec\n'.format(time.time() - start))
        return losses, accuracies

    def _train_step(self, input_seq, target_seq_in, target_seq_out, en_initial_states, optimizer):
        with tf.GradientTape() as tape:
            en_outputs = self.encoder(input_seq, en_initial_states)
            en_states = en_outputs[1:]
            de_states = en_states
            de_outputs = self.decoder(target_seq_in, de_states)
            logits = de_outputs[0]
            loss = self._loss_func(target_seq_out, logits)
            acc = self._accuracy_fn(target_seq_out, logits)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss, acc

    def _loss_func(self, targets, logits):
        crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        mask = tf.math.logical_not(tf.math.equal(targets, 0))
        mask = tf.cast(mask, dtype=tf.int64)
        loss = crossentropy(targets, logits, sample_weight=mask)
        return loss

    def _accuracy_fn(self, y_true, y_pred):
        pred_values = tf.keras.backend.cast(tf.keras.backend.argmax(y_pred, axis=-1), dtype='int64')
        correct = tf.keras.backend.cast(tf.keras.backend.equal(y_true, pred_values), dtype='float32')

        mask = tf.keras.backend.cast(tf.keras.backend.greater(y_true, 0), dtype='float32')
        n_correct = tf.keras.backend.sum(mask * correct)
        n_total = tf.keras.backend.sum(mask)
        return n_correct / n_total


#############################################################################################
# Class: Encoder
#############################################################################################
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(
            hidden_dim, return_sequences=True, return_state=True)

    def call(self, input_sequence, states):
        embed = self.embedding(input_sequence)
        output, state_h, state_c = self.lstm(embed, initial_state=states)
        return output, state_h, state_c

    def init_states(self, batch_size):
        return (tf.zeros([batch_size, self.hidden_dim]),
                tf.zeros([batch_size, self.hidden_dim]))


#############################################################################################
# Class: Decoder
#############################################################################################
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(
            hidden_dim, return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
        self.out = tf.keras.layers.Softmax() 

    def call(self, input_sequence, state):
        embed = self.embedding(input_sequence)
        lstm_out, state_h, state_c = self.lstm(embed, state)
        logits = self.dense(lstm_out)
        out = self.out(logits)
        return logits, state_h, state_c, out