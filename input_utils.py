import os
import tensorflow as tf

from model_utils import Constants as c


# Transform a line of .csv file in a dictionary with features, length and labels.
def read_csv(line, number_of_sensors, number_of_outputs):
    number_of_features_per_interval = c.get_num_features_per_interval(number_of_sensors)
    number_of_features_per_sample = c.get_num_features_per_sample(number_of_sensors)
    defaultVal = [[0.] for idx in range(number_of_features_per_sample + number_of_outputs)] # default values in case of empty columns.
    fileData = tf.decode_csv(line, record_defaults=defaultVal) # Convert CSV records to tensors. Each column maps to one tensor.
    features = fileData[:number_of_features_per_sample]
    features = tf.reshape(features, [c.NUMBER_OF_INTERVALS, number_of_features_per_interval])
    labels = fileData[number_of_features_per_sample:]
    labels = tf.reshape(labels, [number_of_outputs])

    # In the .csv files with the inputs we have that sometimes we don't have all the 20 intervals
    # but to make all the records of the same length they are padded with zeroes at the end.
    # We will then count the exact length of each sample and we will pass it as input, this
    # will then be useful for the RNN layers.
    used = tf.sign(tf.reduce_max(tf.abs(features), reduction_indices=1))
    real_sample_length = tf.reduce_sum(used, reduction_indices=0)
    real_sample_length = tf.cast(real_sample_length, tf.int32)
    
    return {"features":features, "length":real_sample_length}, labels


# Input function: creates an input pipeline that returns a dataset.
def input_fn(data_folder_path, batch_size, number_of_sensors, number_of_outputs, training):
    filename_queue = tf.train.match_filenames_once(os.path.join(data_folder_path, "*.csv"))
        
    dataset = tf.data.TextLineDataset(filename_queue)
    if training: # Shuffle dataset when training
        dataset_len = len(os.listdir(data_folder_path))
        dataset = dataset.shuffle(buffer_size=dataset_len)
    dataset = dataset.map(lambda x: read_csv(x, number_of_sensors, number_of_outputs))
    dataset = dataset.batch(batch_size)
    
    return dataset
    

# Predict Input Function: input function used to predict the output of a single example.
def predict_input_fn(filename, number_of_sensors, number_of_outputs):
        dataset = tf.data.TextLineDataset(filename)
        dataset = dataset.map(lambda x: read_csv(x, number_of_sensors, number_of_outputs))
        return dataset