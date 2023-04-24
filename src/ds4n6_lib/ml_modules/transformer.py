#
# Description: implementation of ML model: seq2seq - Transformer
#

#############################################################################################
# INFO
#############################################################################################

#############################################################################################
# IMPORTS
#############################################################################################
import re, string
import pandas as pd
import numpy as np
from ast import literal_eval
from gensim.models import Word2Vec
from tensorflow import keras
from tensorflow.keras import layers
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
        if type(lm_dset) == str:
            lm_dset = pd.read_csv(lm_dset)
            lm_dset = lm_dset.astype(str)
            lm_dset['path'] = lm_dset['path'].apply(literal_eval)
        lm_dset['time'] = pd.to_datetime(lm_dset['time'], format='%Y-%m-%d')
        lm_dset = lm_dset[(lm_dset['time'] >= from_date) & (lm_dset['time'] <= to_date)]
        model = Word2Vec(list(lm_dset['path']), vector_size=0, min_count=min_count)
        node_list = model.wv.index_to_key
        self.vocab_size = len(node_list) + 5
        
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
        dataset = tf.data.Dataset.from_tensor_slices((train_in, train_out))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(self._format_dataset)
        self.train_dset = dataset.shuffle(len(train_in)).prefetch(16).cache()
    
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
        
    def _format_dataset(self, train_in, train_out):
        t_in = self.in_vectorization(train_in)
        t_out = self.out_vectorization(train_out)
        return ({"encoder_inputs": t_in, "decoder_inputs": t_out[:, :-1],}, t_out[:, 1:])


#############################################################################################
# Class: Autoencoder (Transformer)
#############################################################################################
class Autoencoder:
    def __init__(self, embed_dim, latent_dim, data):
        self.epochs = 5
        self.num_heads = 1
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.data = data
        self.model = None
        
    def set_epochs(self, epochs):
        self.epochs = epochs
    
    def set_num_heads(self, num_heads):
        self.num_heads = num_heads
        
    def build_autoencoder(self):
        encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
        x = PositionalEmbedding(self.data.sequence_length, self.data.vocab_size, self.embed_dim)(encoder_inputs)
        encoder_outputs = TransformerEncoder(self.embed_dim, self.latent_dim, self.num_heads)(x)
        encoder = keras.Model(encoder_inputs, encoder_outputs)

        decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
        encoded_seq_inputs = keras.Input(shape=(None, self.embed_dim), name="decoder_state_inputs")
        x = PositionalEmbedding(self.data.sequence_length, self.data.vocab_size, self.embed_dim)(decoder_inputs)
        x = TransformerDecoder(self.embed_dim, self.latent_dim, self.num_heads)(x, encoded_seq_inputs)
        x = layers.Dropout(0.6)(x)
        decoder_outputs = layers.Dense(self.data.vocab_size, activation="softmax")(x)
        decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

        decoder_outputs = decoder([decoder_inputs, encoder_outputs])
        transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name="transformer")
        self.model = transformer     
        
    def fit_autoencoder(self):
        self.model.summary()
        self.model.compile("rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.model.fit(self.data.train_dset, epochs=self.epochs)
        
    def decode_sequence(self, input_sentence, node_index_dict):
        decoded_err = []
        t_path = input_sentence.split(' ') + (['[eos]']*(self.data.sequence_length))

        tokenized_input_sentence = self.data.in_vectorization([input_sentence])
        decoded_path = '[sos]'
        for i in range(self.data.sequence_length):
            tokenized_target_sentence = self.data.out_vectorization([decoded_path])[:, :-1]
            predictions = self.model([tokenized_input_sentence, tokenized_target_sentence])

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


#############################################################################################
# Class: TransformerEncoder
#############################################################################################
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True
        
    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(query=inputs, value=inputs, key=inputs, attention_mask=padding_mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


#############################################################################################
# Class: PositionalEmbedding
#############################################################################################
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    
#############################################################################################
# Class: TransformerDecoder
#############################################################################################
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential([layers.Dense(latent_dim, activation="relu"), layers.Dense(embed_dim),])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(query=inputs, value=inputs, key=inputs, attention_mask=causal_mask)
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)