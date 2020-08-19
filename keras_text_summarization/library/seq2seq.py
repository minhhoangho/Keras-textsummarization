from __future__ import print_function

from keras.models import Model
from keras.layers import Embedding, Dense, Input, Dropout, \
    Flatten, Multiply, Permute, Activation, TimeDistributed, Add, \
    Bidirectional, Concatenate, Attention
import tensorflow as tf
from keras.optimizers import RMSprop
from keras_text_summarization.library.attention_layer import AttentionLayer
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras_text_summarization.library.utility.glove_loader import load_glove, GLOVE_EMBEDDING_SIZE
import numpy as np
import os

HIDDEN_UNITS = 100
DEFAULT_BATCH_SIZE = 64
VERBOSE = 1
DEFAULT_EPOCHS = 10


class Seq2SeqSummarizer(object):
    model_name = 'seq2seq'

    def __init__(self, config):
        self.num_input_tokens = config['num_input_tokens']
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.input_word2idx = config['input_word2idx']
        self.input_idx2word = config['input_idx2word']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.config = config

        self.version = 0
        if 'version' in config:
            self.version = config['version']

        encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = Embedding(input_dim=self.num_input_tokens, output_dim=HIDDEN_UNITS,
                                      input_length=self.max_input_seq_length, name='encoder_embedding')
        encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = Input(shape=(None, self.num_target_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                         initial_state=encoder_states)
        decoder_dense = Dense(units=self.num_target_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        self.model = model

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)

    def transform_input_text(self, texts):
        temp = []
        for line in texts:
            x = []
            for word in line.lower().split(' '):
                wid = 1
                if word in self.input_word2idx:
                    wid = self.input_word2idx[word]
                x.append(wid)
                if len(x) >= self.max_input_seq_length:
                    break
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_input_seq_length)

        print(temp.shape)
        return temp

    def transform_target_encoding(self, texts):
        temp = []
        for line in texts:
            x = []
            line2 = 'START ' + line.lower() + ' END'
            for word in line2.split(' '):
                x.append(word)
                if len(x) >= self.max_target_seq_length:
                    break
            temp.append(x)

        temp = np.array(temp)
        print(temp.shape)
        return temp

    def generate_batch(self, x_samples, y_samples, batch_size):
        num_batches = len(x_samples) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                encoder_input_data_batch = pad_sequences(x_samples[start:end], self.max_input_seq_length)
                decoder_target_data_batch = np.zeros(
                    shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                decoder_input_data_batch = np.zeros(
                    shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                for lineIdx, target_words in enumerate(y_samples[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in self.target_word2idx:
                            w2idx = self.target_word2idx[w]
                        if w2idx != 0:
                            decoder_input_data_batch[lineIdx, idx, w2idx] = 1
                            if idx > 0:
                                decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
                yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqSummarizer.model_name + '-weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqSummarizer.model_name + '-config.npy'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqSummarizer.model_name + '-architecture.json'

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None, batch_size=None, model_dir_path=None):
        if epochs is None:
            epochs = DEFAULT_EPOCHS
        if model_dir_path is None:
            model_dir_path = './models'
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE

        self.version += 1
        self.config['version'] = self.version
        config_file_path = Seq2SeqSummarizer.get_config_file_path(model_dir_path)
        weight_file_path = Seq2SeqSummarizer.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        np.save(config_file_path, self.config)
        architecture_file_path = Seq2SeqSummarizer.get_architecture_file_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.transform_target_encoding(Ytrain)
        Ytest = self.transform_target_encoding(Ytest)

        Xtrain = self.transform_input_text(Xtrain)
        Xtest = self.transform_input_text(Xtest)

        train_gen = self.generate_batch(Xtrain, Ytrain, batch_size)
        test_gen = self.generate_batch(Xtest, Ytest, batch_size)

        train_num_batches = len(Xtrain) // batch_size
        test_num_batches = len(Xtest) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)
        return history

    def summarize(self, input_text):
        input_seq = []
        input_wids = []
        for word in input_text.lower().split(' '):
            idx = 1  # default [UNK]
            if word in self.input_word2idx:
                idx = self.input_word2idx[word]
            input_wids.append(idx)
        input_seq.append(input_wids)
        input_seq = pad_sequences(input_seq, self.max_input_seq_length)
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_target_tokens))
        target_seq[0, 0, self.target_word2idx['START']] = 1
        target_text = ''
        target_text_len = 0
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = self.target_idx2word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'START' and sample_word != 'END':
                target_text += ' ' + sample_word

            if sample_word == 'END' or target_text_len >= self.max_target_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1, self.num_target_tokens))
            target_seq[0, 0, sample_token_idx] = 1

            states_value = [h, c]
        return target_text.strip()


class Seq2SeqGloVeSummarizer(object):
    model_name = 'seq2seq-glove'

    def __init__(self, config):
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.version = 0
        if 'version' in config:
            self.version = config['version']

        self.word2em = dict()
        if 'unknown_emb' in config:
            self.unknown_emb = config['unknown_emb']
        else:
            self.unknown_emb = np.random.rand(1, GLOVE_EMBEDDING_SIZE)
            config['unknown_emb'] = self.unknown_emb

        self.config = config

        encoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
        encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = Input(shape=(None, self.num_target_tokens), name='decoder_inputs')
        decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                         initial_state=encoder_states)
        decoder_dense = Dense(units=self.num_target_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        self.model = model

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)

    def load_glove(self, data_dir_path):
        self.word2em = load_glove(data_dir_path)

    def transform_input_text(self, texts):
        temp = []
        for line in texts:
            x = np.zeros(shape=(self.max_input_seq_length, GLOVE_EMBEDDING_SIZE))
            for idx, word in enumerate(line.lower().split(' ')):
                if idx >= self.max_input_seq_length:
                    break
                emb = self.unknown_emb
                if word in self.word2em:
                    emb = self.word2em[word]
                x[idx, :] = emb
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_input_seq_length)

        print(temp.shape)
        return temp

    def transform_target_encoding(self, texts):
        temp = []
        for line in texts:
            x = []
            line2 = 'START ' + line.lower() + ' END'
            for word in line2.split(' '):
                x.append(word)
                if len(x) >= self.max_target_seq_length:
                    break
            temp.append(x)

        temp = np.array(temp)
        print(temp.shape)
        return temp

    def generate_batch(self, x_samples, y_samples, batch_size):
        num_batches = len(x_samples) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                encoder_input_data_batch = pad_sequences(x_samples[start:end], self.max_input_seq_length)
                decoder_target_data_batch = np.zeros(
                    shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                decoder_input_data_batch = np.zeros(
                    shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                for lineIdx, target_words in enumerate(y_samples[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in self.target_word2idx:
                            w2idx = self.target_word2idx[w]
                        if w2idx != 0:
                            decoder_input_data_batch[lineIdx, idx, w2idx] = 1
                            if idx > 0:
                                decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
                yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqGloVeSummarizer.model_name + '-weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqGloVeSummarizer.model_name + '-config.npy'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqGloVeSummarizer.model_name + '-architecture.json'

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None, batch_size=None, model_dir_path=None):
        if epochs is None:
            epochs = DEFAULT_EPOCHS
        if model_dir_path is None:
            model_dir_path = './models'
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE

        self.version += 1
        self.config['version'] = self.version
        config_file_path = Seq2SeqGloVeSummarizer.get_config_file_path(model_dir_path)
        weight_file_path = Seq2SeqGloVeSummarizer.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        np.save(config_file_path, self.config)
        architecture_file_path = Seq2SeqGloVeSummarizer.get_architecture_file_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.transform_target_encoding(Ytrain)
        Ytest = self.transform_target_encoding(Ytest)

        Xtrain = self.transform_input_text(Xtrain)
        Xtest = self.transform_input_text(Xtest)

        train_gen = self.generate_batch(Xtrain, Ytrain, batch_size)
        test_gen = self.generate_batch(Xtest, Ytest, batch_size)

        train_num_batches = len(Xtrain) // batch_size
        test_num_batches = len(Xtest) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)
        return history

    def summarize(self, input_text):
        input_seq = np.zeros(shape=(1, self.max_input_seq_length, GLOVE_EMBEDDING_SIZE))
        for idx, word in enumerate(input_text.lower().split(' ')):
            if idx >= self.max_input_seq_length:
                break
            emb = self.unknown_emb  # default [UNK]
            if word in self.word2em:
                emb = self.word2em[word]
            input_seq[0, idx, :] = emb
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_target_tokens))
        target_seq[0, 0, self.target_word2idx['START']] = 1
        target_text = ''
        target_text_len = 0
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = self.target_idx2word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'START' and sample_word != 'END':
                target_text += ' ' + sample_word

            if sample_word == 'END' or target_text_len >= self.max_target_seq_length:
                terminated = True

            target_seq = np.zeros((1, 1, self.num_target_tokens))
            target_seq[0, 0, sample_token_idx] = 1

            states_value = [h, c]
        return target_text.strip()


class Seq2SeqGloVeSummarizerV2(object):
    model_name = 'seq2seq-glove-v2'

    def __init__(self, config):
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.version = 0
        if 'version' in config:
            self.version = config['version']

        self.word2em = dict()
        if 'unknown_emb' in config:
            self.unknown_emb = config['unknown_emb']
        else:
            self.unknown_emb = np.random.rand(1, GLOVE_EMBEDDING_SIZE)
            config['unknown_emb'] = self.unknown_emb

        self.config = config

        encoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
        encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')
        decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                         initial_state=encoder_states)
        decoder_dense = Dense(units=self.num_target_tokens, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        self.model = model

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)

    def load_glove(self, data_dir_path):
        self.word2em = load_glove(data_dir_path)

    def transform_input_text(self, texts):
        temp = []
        for line in texts:
            x = np.zeros(shape=(self.max_input_seq_length, GLOVE_EMBEDDING_SIZE))
            for idx, word in enumerate(line.lower().split(' ')):
                if idx >= self.max_input_seq_length:
                    break
                emb = self.unknown_emb
                if word in self.word2em:
                    emb = self.word2em[word]
                x[idx, :] = emb
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_input_seq_length)

        print(temp.shape)
        return temp

    def transform_target_encoding(self, texts):
        temp = []
        for line in texts:
            x = []
            line2 = 'start ' + line.lower() + ' end'
            for word in line2.split(' '):
                x.append(word)
                if len(x) >= self.max_target_seq_length:
                    break
            temp.append(x)

        temp = np.array(temp)
        print(temp.shape)
        return temp

    def generate_batch(self, x_samples, y_samples, batch_size):
        num_batches = len(x_samples) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                encoder_input_data_batch = pad_sequences(x_samples[start:end], self.max_input_seq_length)
                decoder_target_data_batch = np.zeros(
                    shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                decoder_input_data_batch = np.zeros(
                    shape=(batch_size, self.max_target_seq_length, GLOVE_EMBEDDING_SIZE))
                for lineIdx, target_words in enumerate(y_samples[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in self.word2em:
                            emb = self.unknown_emb
                            decoder_input_data_batch[lineIdx, idx, :] = emb
                        if w in self.target_word2idx:
                            w2idx = self.target_word2idx[w]
                        if w2idx != 0:
                            if idx > 0:
                                decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
                yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqGloVeSummarizerV2.model_name + '-weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqGloVeSummarizerV2.model_name + '-config.npy'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqGloVeSummarizerV2.model_name + '-architecture.json'

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None, batch_size=None, model_dir_path=None):
        if epochs is None:
            epochs = DEFAULT_EPOCHS
        if model_dir_path is None:
            model_dir_path = './models'
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE

        self.version += 1
        self.config['version'] = self.version
        config_file_path = Seq2SeqGloVeSummarizerV2.get_config_file_path(model_dir_path)
        weight_file_path = Seq2SeqGloVeSummarizerV2.get_weight_file_path(model_dir_path)

        checkpoint = ModelCheckpoint(weight_file_path)
        np.save(config_file_path, self.config)
        architecture_file_path = Seq2SeqGloVeSummarizerV2.get_architecture_file_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.transform_target_encoding(Ytrain)
        Ytest = self.transform_target_encoding(Ytest)

        Xtrain = self.transform_input_text(Xtrain)
        Xtest = self.transform_input_text(Xtest)

        train_gen = self.generate_batch(Xtrain, Ytrain, batch_size)
        test_gen = self.generate_batch(Xtest, Ytest, batch_size)

        train_num_batches = len(Xtrain) // batch_size
        test_num_batches = len(Xtest) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)
        return history

    def summarize(self, input_text):
        input_seq = np.zeros(shape=(1, self.max_input_seq_length, GLOVE_EMBEDDING_SIZE))
        for idx, word in enumerate(input_text.lower().split(' ')):
            if idx >= self.max_input_seq_length:
                break
            emb = self.unknown_emb  # default [UNK]
            if word in self.word2em:
                emb = self.word2em[word]
            input_seq[0, idx, :] = emb
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, GLOVE_EMBEDDING_SIZE))
        target_seq[0, 0, :] = self.word2em['start']
        target_text = ''
        target_text_len = 0
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = self.target_idx2word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'start' and sample_word != 'end':
                target_text += ' ' + sample_word

            if sample_word == 'end' or target_text_len >= self.max_target_seq_length:
                terminated = True

            if sample_word in self.word2em:
                target_seq[0, 0, :] = self.word2em[sample_word]
            else:
                target_seq[0, 0, :] = self.unknown_emb

            states_value = [h, c]
        return target_text.strip()


class Seq2SeqGloVeSummarizerV3(object):
    model_name = 'seq2seq-glove-v3'

    def __init__(self, config):
        self.max_input_seq_length = config['max_input_seq_length']
        self.num_target_tokens = config['num_target_tokens']
        self.max_target_seq_length = config['max_target_seq_length']
        self.target_word2idx = config['target_word2idx']
        self.target_idx2word = config['target_idx2word']
        self.version = 0
        if 'version' in config:
            self.version = config['version']

        self.word2em = dict()
        if 'unknown_emb' in config:
            self.unknown_emb = config['unknown_emb']
        else:
            self.unknown_emb = np.random.rand(1, GLOVE_EMBEDDING_SIZE)
            config['unknown_emb'] = self.unknown_emb

        self.config = config

        # """__encoder___"""
        # encoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
        # encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm', dropout=0.2)
        # encoder_lstm_rev = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, dropout=0.05,
        #                         go_backwards=True)
        # encoder_embedding = Embedding(input_dim=self.num_input_tokens, output_dim=HIDDEN_UNITS,
        #                               input_length=self.max_input_seq_length, name='encoder_embedding')

        # encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
        # encoder_outputs_rev, encoder_state_h_rev, encoder_state_c_rev = encoder_lstm_rev(encoder_inputs)

        # encoder_state_h_final = Add()(([encoder_state_h, encoder_state_h_rev]))
        # encoder_state_c_final = Add()([encoder_state_c, encoder_state_c_rev])
        # encoder_outputs_final = Add()([encoder_outputs, encoder_outputs_rev])

        # encoder_states = [encoder_state_h_final, encoder_state_c_final]

        # """____decoder___"""
        # decoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')
        # decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm',
        #                     dropout=0.2)
        #
        # decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
        #                                                                  initial_state=encoder_states)
        # # Pull out XGBoost, (Attention)
        # attention = TimeDistributed(Dense(1, activation='tanh'))(encoder_outputs_final)
        # attention = Flatten()(attention)
        # attention = Multiply()([decoder_outputs, attention])
        # attention = Activation('softmax')(attention)
        # attention = Permute([1, 2])(attention)
        # print(f'attention layer {attention}')
        #
        # decoder_dense = Dense(units=self.num_target_tokens, activation='softmax', name='decoder_dense')
        # # decoder_dense = Dense(units=self.num_target_tokens, activation='linear', name='decoder_dense')
        # decoder_outputs = decoder_dense(attention)
        # # decoder_outputs = decoder_dense(decoder_outputs)
        # model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        # rmsprop = RMSprop(lr=0.005, clipnorm=2.0)
        # # model.compile(loss='mse', optimizer=rmsprop, metrics=['accuracy'])
        # model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])
        # self.model = model
        # print(model.summary())
        #
        # """_________________inference mode__________________"""
        #
        # self.encoder_model = Model(encoder_inputs, encoder_states)
        #
        # decoder_state_input_h = Input(shape=(HIDDEN_UNITS,))
        # decoder_state_input_c = Input(shape=(HIDDEN_UNITS,))
        #
        # decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
        # decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        # decoder_states = [state_h, state_c]
        # decoder_outputs = decoder_dense(decoder_outputs)
        # self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

        """__encoder___"""
        # GRU 1
        encoder_inputs = tf.keras.layers.Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
        encoder_gru_01 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units=HIDDEN_UNITS, return_sequences=True, return_state=True))
        encoder_output_01, encoder_forward_state_01, encoder_backward_state_01 = encoder_gru_01(encoder_inputs)
        encoder_output_dropout_01 = tf.keras.layers.Dropout(0.3)(encoder_output_01)
        # GRU 2
        encoder_gru_02 = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units=HIDDEN_UNITS, return_sequences=True, return_state=True))
        encoder_output, encoder_forward_state, encoder_backward_state = encoder_gru_02(encoder_output_dropout_01)
        encoder_state = tf.keras.layers.Concatenate()([encoder_forward_state, encoder_backward_state])

        """__decoder__"""
        decoder_inputs = tf.keras.layers.Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')
        # GRU using encoder_states as initial state
        decoder_gru = tf.keras.layers.GRU(HIDDEN_UNITS * 2, return_sequences=True, return_state=True)
        decoder_output, decoder_state = decoder_gru(decoder_inputs, initial_state=[encoder_state])

        # Attention Layer
        attention_layer = AttentionLayer()
        attention_out, attention_states = attention_layer([encoder_output, decoder_output])

        # Concat attention output and decoder GRU output
        decoder_concatenate = tf.keras.layers.Concatenate(axis=-1)([decoder_output, attention_out])

        # Dense layer
        decoder_dense = tf.keras.layers.TimeDistributed(
            Dense(self.num_target_tokens, activation='softmax'))  # hierarchical
        decoder_dense_output = decoder_dense(decoder_concatenate)
        # Define the model
        model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_dense_output)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        print(model.summary())

        """_________________inference mode__________________"""
        # Encoder Inference Model
        self.encoder_model = tf.keras.Model(encoder_inputs, [encoder_output, encoder_state])

        # Decoder Inference
        # Below tensors will hold the states of the previous time step
        decoder_state = tf.keras.layers.Input(shape=(HIDDEN_UNITS * 2,))
        decoder_intermittent_state_input = tf.keras.layers.Input(shape=(self.max_input_seq_length, HIDDEN_UNITS * 2))

        # Get Embeddings of Decoder Sequence
        # decoder_embedding_inference = embedding_layer_headline(decoder_input)

        # Predict Next Word in Sequence, Set Initial State to State from Previous Time Step
        decoder_output_inference, decoder_state_inference = decoder_gru(decoder_inputs,
                                                                        initial_state=[decoder_state])

        # Attention Inference
        attention_layer = AttentionLayer()
        attention_out_inference, attention_state_inference = attention_layer(
            [decoder_intermittent_state_input, decoder_output_inference])
        decoder_inference_concat = tf.keras.layers.Concatenate(axis=-1)(
            [decoder_output_inference, attention_out_inference])

        # Dense Softmax Layer to Generate Prob. Dist. Over Target Vocabulary
        decoder_output_inference = decoder_dense(decoder_inference_concat)

        # Final Decoder Model
        self.decoder_model = tf.keras.Model([decoder_inputs, decoder_intermittent_state_input, decoder_state],
                                            [decoder_output_inference, decoder_state_inference])

    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)

    def load_glove(self, data_dir_path):
        self.word2em = load_glove(data_dir_path)

    def transform_input_text(self, texts):
        temp = []
        for line in texts:
            x = np.zeros(shape=(self.max_input_seq_length, GLOVE_EMBEDDING_SIZE))
            for idx, word in enumerate(line.lower().split(' ')):
                if idx >= self.max_input_seq_length:
                    break
                emb = self.unknown_emb
                if word in self.word2em:
                    emb = self.word2em[word]
                x[idx, :] = emb
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_input_seq_length)
        print(f'Transform input text shape: {temp.shape}')
        return temp

    def transform_target_encoding(self, texts):
        temp = []
        for line in texts:
            x = []
            line2 = 'start ' + line.lower() + ' end'
            for word in line2.split(' '):
                x.append(word)
                if len(x) >= self.max_target_seq_length:
                    break
            temp.append(x)
        temp = np.array(temp)
        print(f'Transform target encoding shape: {temp.shape}')
        return temp

    @staticmethod
    def get_weight_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqGloVeSummarizerV3.model_name + '-weights.h5'

    @staticmethod
    def get_config_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqGloVeSummarizerV3.model_name + '-config.npy'

    @staticmethod
    def get_architecture_file_path(model_dir_path):
        return model_dir_path + '/' + Seq2SeqGloVeSummarizerV3.model_name + '-architecture.json'

    def generate_batch(self, x_samples, y_samples, batch_size):
        num_batches = len(x_samples) // batch_size
        while True:
            for batchIdx in range(0, num_batches):
                start = batchIdx * batch_size
                end = (batchIdx + 1) * batch_size
                encoder_input_data_batch = pad_sequences(x_samples[start:end], self.max_input_seq_length)
                decoder_target_data_batch = np.zeros(
                    shape=(batch_size, self.max_target_seq_length, self.num_target_tokens))
                decoder_input_data_batch = np.zeros(
                    shape=(batch_size, self.max_target_seq_length, GLOVE_EMBEDDING_SIZE))
                for lineIdx, target_words in enumerate(y_samples[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in self.word2em:
                            emb = self.unknown_emb
                            decoder_input_data_batch[lineIdx, idx, :] = emb
                        if w in self.target_word2idx:
                            w2idx = self.target_word2idx[w]
                        if w2idx != 0:
                            if idx > 0:
                                decoder_target_data_batch[lineIdx, idx - 1, w2idx] = 1
                yield [encoder_input_data_batch, decoder_input_data_batch], decoder_target_data_batch

    def encode_data(self, X, Y):
        # print(f'X encode shape: {X.shape}')
        # print(f'Y encode shape: {Y.shape}')
        if X.shape[0] != Y.shape[0]:
            print('X, Y shape for encode is not equal, exit ...')
            exit(0)
        encoder_input = pad_sequences(X, self.max_input_seq_length)
        decoder_target = np.zeros(
            shape=(Y.shape[0], self.max_target_seq_length, self.num_target_tokens))
        decoder_input = np.zeros(
            shape=(Y.shape[0], self.max_target_seq_length, GLOVE_EMBEDDING_SIZE))
        for lineIdx, target_words in enumerate(Y):
            for idx, w in enumerate(target_words):
                w2idx = 0  # default [UNK]
                if w in self.word2em:
                    emb = self.unknown_emb
                    decoder_input[lineIdx, idx, :] = emb
                if w in self.target_word2idx:
                    w2idx = self.target_word2idx[w]
                if w2idx != 0:
                    if idx > 0:
                        decoder_target[lineIdx, idx - 1, w2idx] = 1
        # print(f'encoder_input shape: {np.shape(encoder_input)}')
        # print(f'decoder_input shape: {np.shape(decoder_input)}')
        # print(f'decoder_target shape: {np.shape(decoder_target)}')
        # print("-------------------------")
        return encoder_input, decoder_input, decoder_target

    def fit(self, Xtrain, Ytrain, Xtest, Ytest, epochs=None, batch_size=None, model_dir_path=None):
        if epochs is None:
            epochs = DEFAULT_EPOCHS
        if model_dir_path is None:
            model_dir_path = './models'
        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE

        self.version += 1
        self.config['version'] = self.version
        config_file_path = Seq2SeqGloVeSummarizerV3.get_config_file_path(model_dir_path)
        weight_file_path = Seq2SeqGloVeSummarizerV3.get_weight_file_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        np.save(config_file_path, self.config)
        architecture_file_path = Seq2SeqGloVeSummarizerV3.get_architecture_file_path(model_dir_path)
        print(f'architecture {architecture_file_path}')
        open(architecture_file_path, 'w').write(self.model.to_json())

        Ytrain = self.transform_target_encoding(Ytrain)
        Ytest = self.transform_target_encoding(Ytest)

        Xtrain = self.transform_input_text(Xtrain)
        Xtest = self.transform_input_text(Xtest)

        Xtrain_inp, Ytrain_inp, Ytrain_target = self.encode_data(Xtrain, Ytrain)
        Xtest_inp, Ytest_inp, Ytest_target = self.encode_data(Xtest, Ytest)
        # train_gen = self.generate_batch(Xtrain, Ytrain, batch_size)
        # test_gen = self.generate_batch(Xtest, Ytest, batch_size)
        #
        # train_num_batches = len(Xtrain) // batch_size
        # test_num_batches = len(Xtest) // batch_size

        history = self.model.fit([Xtrain_inp, Ytrain_inp], Ytrain_target, batch_size=batch_size, epochs=epochs,
                                 verbose=VERBOSE,
                                 validation_data=([Xtest_inp, Ytest_inp], Ytest_target))
        # history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
        #                                    epochs=epochs,
        #                                    verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
        #                                    callbacks=[checkpoint])
        # scores = self.model.evaluate([Xtest, Ytest], Ytest, verbose=1)
        self.model.save_weights(weight_file_path)
        return history

    def summarize(self, input_text):
        input_seq = np.zeros(shape=(1, self.max_input_seq_length, GLOVE_EMBEDDING_SIZE))
        for idx, word in enumerate(input_text.lower().split(' ')):
            if idx >= self.max_input_seq_length:
                break
            emb = self.unknown_emb  # default [UNK]
            if word in self.word2em:
                emb = self.word2em[word]
            input_seq[0, idx, :] = emb
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, GLOVE_EMBEDDING_SIZE))
        target_seq[0, 0, :] = self.word2em['start']
        target_text = ''
        target_text_len = 0
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = self.target_idx2word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'start' and sample_word != 'end':
                target_text += ' ' + sample_word

            if sample_word == 'end' or target_text_len >= self.max_target_seq_length:
                terminated = True

            if sample_word in self.word2em:
                target_seq[0, 0, :] = self.word2em[sample_word]
            else:
                target_seq[0, 0, :] = self.unknown_emb

            states_value = [h, c]
        return target_text.strip()
