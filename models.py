import os
import tensorflow as tf

import model_layers as ml
import model_utils as mu
from model_utils import Constants as c


class Model():
    def __init__(self, number_of_sensors, number_of_output_classes, hyper_params):
        self.num_sensors = number_of_sensors
        self.num_outputs = number_of_output_classes
        self.hyer_params = hyper_params
        
    def get_model_function(self):
        def model_fn(features, labels, mode, params):
            return self._model_function(features, labels, mode, params)
        return model_fn
        
    def _model_function(self, features, labels, mode, params):
        model_output, tl_input = self._create_model(features, mode, params)
    
        # PREDICTION MODE
        if mode == tf.estimator.ModeKeys.PREDICT:
            return mu.get_predict_estimatorspec(model_output, tl_input)

        # Calculate the loss
        ce_loss = mu.cross_entropy_loss(model_output, labels)
        l2_loss = mu.l2_regularization_loss(self.hyer_params["l2_lambda_term"])
        loss = ce_loss + l2_loss
        
        # EVALUATION MODE
        if mode == tf.estimator.ModeKeys.EVAL:
            return mu.get_eval_estimatorspec(model_output, labels, loss, self.num_outputs)
    
        # TRAINING MODE
        if mode == tf.estimator.ModeKeys.TRAIN:
            return mu.get_training_estimatorspec(loss, self.hyer_params)
        
    def _create_model(self, features, mode, params):
        # Each subclass defines this method. Here you create the tensorflow computation graph and
        # return the output, and the input of the last layer (to be used for personalization).
        pass


class LSTM(Model):
    def _create_model(self, features, mode, params):  
        num_intervals = c.NUMBER_OF_INTERVALS
        feat_dim = c.get_num_features_per_interval(self.num_sensors)            
        sensor_inputs, length = mu.prepare_features(features, num_intervals, feat_dim, mode)
        
        inputs_shape = tf.shape(sensor_inputs) # (BATCH_SIZE, NUMBER_OF_INTERVALS, FEATURE_DIM, CHANNEL=1)
        sensor_inputs = tf.reshape(sensor_inputs, [inputs_shape[0], num_intervals, feat_dim]) # Get rid of channel dimension
        batch_size = inputs_shape[0]
    
        #------ RNN Layers
        num_cells = 256
        lstm_cell1 = tf.contrib.rnn.LSTMCell(num_cells)
        if mode == tf.estimator.ModeKeys.TRAIN:
            lstm_cell1 = tf.contrib.rnn.DropoutWrapper(lstm_cell1, output_keep_prob=0.5)

        lstm_cell2 = tf.contrib.rnn.LSTMCell(num_cells)
        if mode == tf.estimator.ModeKeys.TRAIN:
            lstm_cell2 = tf.contrib.rnn.DropoutWrapper(lstm_cell2, output_keep_prob=0.5)

        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell1, lstm_cell2])
        init_state = cell.zero_state(batch_size, tf.float32)
    
        cell_output, final_stateTuple = tf.nn.dynamic_rnn(cell, sensor_inputs, sequence_length=length, initial_state=init_state, time_major=False)
        # cell_output has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, number_of_cells).

        cell_output = tf.Print(cell_output, [tf.shape(cell_output), tf.shape(final_stateTuple)])

        # Sum the output of the RNN for each example and calculate the mean.
        sum_cell_out = tf.reduce_sum(cell_output, axis=1, keepdims=False)
        l = tf.reshape(length, [batch_size, 1]) 
        l = tf.cast(l, tf.float32)
        avg_cell_out = sum_cell_out/(tf.tile(l, [1, num_cells])) # we have to calculate the mean this way to take into account for the different lengths.

        #------ Output Layer
        logits = ml.output_layer(avg_cell_out, self.num_outputs)
    
        return logits, avg_cell_out


class DeepSense(Model):
    def _create_model(self, features, mode, params):        
        sensor_inputs, length = mu.prepare_features(features, c.NUMBER_OF_INTERVALS, c.get_num_features_per_interval(self.num_sensors), mode)
        batch_size = tf.shape(sensor_inputs)[0]
    
        # Separate sensors data.
        num_sensors = self.num_sensors
        separate_sensors_data = tf.split(sensor_inputs, num_or_size_splits=num_sensors, axis=2)
        
        #------ Individual Convolutional Layers    & Merge Convolutional Layers
        conv_layers_output = ml.ds_convolutional_layer(separate_sensors_data, num_sensors, mode)

        # Reshape for Recurrent Neural Network.
        clo_shape = tf.shape(conv_layers_output)
        conv_layers_output = tf.reshape(conv_layers_output, [-1, clo_shape[1], num_sensors*4*64])
        # sensor_conv_out has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, ...)
    
        #------ RNN Layers
        num_cells = 120
        cell_output = ml.stacked_gru_layer(conv_layers_output, length, num_cells, mode)
        # cell_output has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, num_cells=120).

        # Sum the output of the RNN for each example and calculate the mean.
        sum_cell_out = tf.reduce_sum(cell_output, axis=1, keepdims=False)
        l = tf.reshape(length, [batch_size, 1]) 
        l = tf.cast(l, tf.float32)
        avg_cell_out = sum_cell_out/(tf.tile(l, [1, num_cells])) # we have to calculate the mean this way to take into account for the different lengths.

        #------ Output Layer
        logits = ml.output_layer(avg_cell_out, self.num_outputs)
    
        return logits, avg_cell_out
        

class SADeepSense(Model):
    def _create_model(self, features, mode, params):        
        sensor_inputs, length = mu.prepare_features(features, c.NUMBER_OF_INTERVALS, c.get_num_features_per_interval(self.num_sensors), mode)
    
        # Separate sensors data.
        num_sensors = self.num_sensors
        separate_sensors_data = tf.split(sensor_inputs, num_or_size_splits=num_sensors, axis=2)
        
        #------ Individual Convolutional Layers    & Merge Convolutional Layers
        conv_layers_output = ml.sads_convolutional_layer(separate_sensors_data, self.num_sensors, self.num_outputs, mode)

        # Reshape for Recurrent Neural Network.
        clo_shape = tf.shape(conv_layers_output)
        conv_layers_output = tf.reshape(conv_layers_output, [-1, clo_shape[1], 4*64])
        # sensor_conv_out has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, ...)
    
        #------ RNN Layers
        num_cells = 120
        cell_output = ml.stacked_gru_layer(conv_layers_output, length, num_cells, mode)
        # cell_output has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, num_cells=120).
        
        #------ Temporal Self-Attention Module
        merged_timesteps = ml.sa_temporal_module(cell_output, c.NUMBER_OF_INTERVALS, self.num_outputs)

        #------ Output Layer
        logits = ml.output_layer(merged_timesteps, self.num_outputs)
    
        return logits, merged_timesteps
        

class TrASenD_BD(Model):
    def _create_model(self, features, mode, params):
        sensor_inputs, length = mu.prepare_features(features, c.NUMBER_OF_INTERVALS, c.get_num_features_per_interval(self.num_sensors), mode)
        batch_size = tf.shape(sensor_inputs)[0]
    
        # Separate sensors data.
        num_sensors = self.num_sensors
        separate_sensors_data = tf.split(sensor_inputs, num_or_size_splits=num_sensors, axis=2)
        
        #------ Individual Convolutional Layers    & Merge Convolutional Layers
        conv_layers_output = ml.ds_convolutional_layer(separate_sensors_data, num_sensors, mode)

        # Reshape for Recurrent Neural Network.
        clo_shape = tf.shape(conv_layers_output)
        conv_layers_output = tf.reshape(conv_layers_output, [-1, clo_shape[1], num_sensors*4*64])
        # sensor_conv_out has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, ...)
    
        #------ RNN Layers
        num_cells = 120
        cell_output = ml.bidirectional_gru_layer(conv_layers_output, length, num_cells, mode)
        # cell_output has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, 120).

        # Sum the output of the RNN for each example and calculate the mean.
        sum_cell_out = tf.reduce_sum(cell_output, axis=1, keepdims=False)
        l = tf.reshape(length, [batch_size, 1]) 
        l = tf.cast(l, tf.float32)
        avg_cell_out = sum_cell_out/(tf.tile(l, [1, num_cells])) # we have to calculate the mean this way to take into account for the different lengths.

        #------ Output Layer
        logits = ml.output_layer(avg_cell_out, self.num_outputs)
    
        return logits, avg_cell_out
        

class TrASenD_CA(Model):
    def _create_model(self, features, mode, params):
        sensor_inputs, length = mu.prepare_features(features, c.NUMBER_OF_INTERVALS, c.get_num_features_per_interval(self.num_sensors), mode)
        batch_size = tf.shape(sensor_inputs)[0]
    
        # Separate sensors data.
        num_sensors = self.num_sensors
        separate_sensors_data = tf.split(sensor_inputs, num_or_size_splits=num_sensors, axis=2)
        
        #------ Individual Convolutional Layers    & Merge Convolutional Layers
        conv_layers_output = ml.ds_convolutional_layer(separate_sensors_data, num_sensors, mode)

        # Reshape for Attention Mechanism
        clo_shape = tf.shape(conv_layers_output)
        conv_layers_output = tf.reshape(conv_layers_output, [-1, clo_shape[1], num_sensors*4, 64])
        conv_layers_output = tf.transpose(conv_layers_output, perm=[0, 1, 3, 2])
        # conv_layers_output has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, CHANNELS, FEATURES=clo_shape[2]*clo_shape[3])
        
        #------ GRU with attention mechanism
        num_cells = 120
        att_gru_output = ml.attention_gru(conv_layers_output, length, num_cells)
        
        # Sum the output at each timestep, taking sequence lengths into account
        sum_timesteps_output = tf.reduce_sum(att_gru_output, axis=1, keepdims=False)
        l = tf.reshape(length, [batch_size, 1]) 
        l = tf.cast(l, tf.float32)
        avg_cell_out = sum_timesteps_output/(tf.tile(l, [1, num_cells]))
    
        #------ Output Layer
        logits = ml.output_layer(sum_timesteps_output, self.num_outputs)
        
        return logits, att_gru_output
        

class TrASenD(Model):
    def _create_model(self, features, mode, params):
        sensor_inputs, length = mu.prepare_features(features, c.NUMBER_OF_INTERVALS, c.get_num_features_per_interval(self.num_sensors), mode)
        batch_size = tf.shape(sensor_inputs)[0]
    
        # Separate sensors data.
        num_sensors = self.num_sensors
        separate_sensors_data = tf.split(sensor_inputs, num_or_size_splits=num_sensors, axis=2)
        
        #------ Individual Convolutional Layers    & Merge Convolutional Layers
        conv_layers_output = ml.ds_convolutional_layer(separate_sensors_data, num_sensors, mode)

        # Reshape for Attention Mechanism
        clo_shape = tf.shape(conv_layers_output)
        conv_layers_output = tf.reshape(conv_layers_output, [-1, clo_shape[1], num_sensors*4*64])
        # conv_layers_output has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, FEATURES*CHANNELS)

        #------ Transformer Encoder
        transformer_output = ml.transformer_encoder(conv_layers_output)
        transformer_output = tf.reshape(transformer_output, (clo_shape[0], c.NUMBER_OF_INTERVALS* transformer_output.get_shape().as_list()[2]))
        
        #------ Output Layer
        logits = ml.output_layer(transformer_output, self.num_outputs)
        
        return logits, transformer_output