import tensorflow as tf


def individual_conv_layers(layer_input, num_sensors, mode):
    conv1 = tf.layers.conv2d(layer_input, filters=64, kernel_size=[1, 6*3], strides=[1, 6], padding="valid")
    conv1 = tf.layers.batch_normalization(conv1, training=(mode == tf.estimator.ModeKeys.TRAIN))
    conv1 = tf.nn.relu(conv1)
    conv1_shape = tf.shape(conv1) # Use this to define the shape of the dropout mask.
    conv1 = tf.layers.dropout(conv1, rate=0.2, 
        noise_shape=[conv1_shape[0], 1, 1, conv1_shape[3]], training=(mode == tf.estimator.ModeKeys.TRAIN))

    conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=[1, 3], strides=[1, 1], padding="valid")
    conv2 = tf.layers.batch_normalization(conv2, training=(mode == tf.estimator.ModeKeys.TRAIN))
    conv2 = tf.nn.relu(conv2)
    conv2_shape = tf.shape(conv2)
    conv2 = tf.layers.dropout(conv2, rate=0.2,
        noise_shape=[conv2_shape[0], 1, 1, conv2_shape[3]], training=(mode == tf.estimator.ModeKeys.TRAIN))

    conv3 = tf.layers.conv2d(conv2, filters=64, kernel_size=[1, 3], strides=[1, 1], padding="valid")
    conv3 = tf.layers.batch_normalization(conv3, training=(mode == tf.estimator.ModeKeys.TRAIN))
    conv3 = tf.nn.relu(conv3)
    
    return conv3
    
    
def merge_conv_layer(layer_input, num_sensors, mode):
    conv1 = tf.layers.conv3d(layer_input, filters=64, kernel_size=[1, num_sensors, 8], strides=[1, 1, 1], padding="same")
    conv1 = tf.layers.batch_normalization(conv1, training=(mode == tf.estimator.ModeKeys.TRAIN))
    conv1 = tf.nn.relu(conv1)
    conv1_shape = tf.shape(conv1)
    conv1 = tf.layers.dropout(conv1, rate=0.2,
        noise_shape=[conv1_shape[0], 1, 1, 1, conv1_shape[4]], training=(mode == tf.estimator.ModeKeys.TRAIN))
        
    conv2 = tf.layers.conv3d(conv1, filters=64, kernel_size=[1, num_sensors, 6], strides=[1, 1, 1], padding="same")
    conv2 = tf.layers.batch_normalization(conv2, training=(mode == tf.estimator.ModeKeys.TRAIN))
    conv2 = tf.nn.relu(conv2)
    conv2_shape = tf.shape(conv2)
    conv2 = tf.layers.dropout(conv2, rate=0.2,
        noise_shape=[conv2_shape[0], 1, 1, 1, conv2_shape[4]], training=(mode == tf.estimator.ModeKeys.TRAIN))
        
    conv3 = tf.layers.conv3d(conv2, filters=64, kernel_size=[1, num_sensors, 4], strides=[1, 1, 1], padding="same")
    conv3 = tf.layers.batch_normalization(conv3, training=(mode == tf.estimator.ModeKeys.TRAIN))
    conv3 = tf.nn.relu(conv3)
    
    return conv3
    

def ds_convolutional_layer(sensors_input, num_sensors, mode):
    #------ Individual Convolutional Layers
    sensors_indiv_layer_out = []
    for sensor_data in sensors_input:
        individual_layer_out = individual_conv_layers(sensor_data, num_sensors, mode)
        # Reshape for future concatenation    
        individual_layer_out_shape = tf.shape(individual_layer_out)
        individual_layer_out = tf.reshape(individual_layer_out, [-1, individual_layer_out_shape[1], 1, individual_layer_out_shape[2], individual_layer_out_shape[3]])
        sensors_indiv_layer_out.append(individual_layer_out)

    # Concatenate the output of the individual convolutional layers then apply dropout.
    sensor_conv_in = tf.concat(sensors_indiv_layer_out, 2)
    # sensor_conv_in has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, num_sensors, ...(depends on kernel_size and padding of conv layers), CHANNELS).
    senor_conv_shape_l = sensor_conv_in.get_shape().as_list()
    senor_conv_shape = tf.shape(sensor_conv_in)
    sensor_conv_in = tf.layers.dropout(sensor_conv_in, rate=0.2,
        noise_shape=[senor_conv_shape[0], 1, 1, 1, senor_conv_shape[4]], training=(mode == tf.estimator.ModeKeys.TRAIN))
    sensor_conv_in.set_shape([None, senor_conv_shape_l[1], num_sensors, senor_conv_shape_l[3], 64])
    
    #------ Merge Convolutional Layers    
    merge_conv_output = merge_conv_layer(sensor_conv_in, num_sensors, mode)
    
    return merge_conv_output
    

def sa_conv_module(layer_input, num_sensors, number_of_outputs):
    h = number_of_outputs
    # Apply transformation f -> f_transf_output has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, num_sensors, ..., h)
    f_transf_output = tf.layers.conv3d(layer_input, filters=h, kernel_size=[1, 1, 1], strides=[1, 1, 1], padding="valid")
    # Apply transformation g -> g_transf_output has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, 1, ..., h)
    g_transf_output = tf.layers.conv3d(layer_input, filters=h, kernel_size=[1, num_sensors, 1], strides=[1, 1, 1], padding="valid")
    
    # Calculate correlation heatmap M -> M has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, num_sensors, ..., h)
    M = tf.multiply(f_transf_output, g_transf_output)
    M = tf.multiply(M, tf.pow(tf.constant(h, dtype=tf.float32), tf.constant(-1/2)))
    
    # Calculate attention weight -> w has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, num_sensors)
    w = tf.reduce_mean(M, axis=-1)
    w = tf.reduce_mean(w, axis=-1)
    w = tf.nn.softmax(w)
    
    # Apply weights & "merge"
    weighted_input = tf.einsum('bntfc,bnt->bntfc', layer_input, w)
    output = tf.reduce_sum(weighted_input, axis=2)
    return output
    
    
def sa_temporal_module(layer_input, num_intervals, number_of_outputs):
    h = number_of_outputs
    layer_input_expanded = tf.expand_dims(layer_input, axis=-1)
    # Apply transformation f -> f_transf_output has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, cell_output_dim, h)
    f_transf_output = tf.layers.conv2d(layer_input_expanded, filters=h, kernel_size=[1, 1], strides=[1, 1], padding="valid")
    # Apply transformation g -> g_transf_output has shape (BATCH_SIZE, 1, cell_output_dim, h)
    g_transf_output = tf.layers.conv2d(layer_input_expanded, filters=h, kernel_size=[num_intervals, 1], strides=[1, 1], padding="valid")
    
    # Calculate correlation heatmap M -> M has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, cell_output_dim, h)
    M = tf.multiply(f_transf_output, g_transf_output)
    M = tf.multiply(M, tf.pow(tf.constant(h, dtype=tf.float32), tf.constant(-1/2)))
    
    # Calculate attention weight -> w has shape (BATCH_SIZE, NUMBER_OF_INTERVALS)
    w = tf.reduce_mean(M, axis=-1)
    w = tf.reduce_mean(w, axis=-1)
    w = tf.nn.softmax(w)
    
    # Apply weights & "merge"
    weighted_input = tf.einsum('bnf,bn->bnf', layer_input, w)
    output = tf.reduce_sum(weighted_input, axis=1)
    return output


def sads_convolutional_layer(sensors_input, num_sensors, number_of_outputs, mode):
    #------ Individual Convolutional Layers
    sensors_indiv_layer_out = []
    for sensor_data in sensors_input:
        individual_layer_out = individual_conv_layers(sensor_data, num_sensors, mode)
        # Reshape for future concatenation    
        individual_layer_out_shape = tf.shape(individual_layer_out)
        individual_layer_out = tf.reshape(individual_layer_out, [-1, individual_layer_out_shape[1], 1, individual_layer_out_shape[2], individual_layer_out_shape[3]])
        sensors_indiv_layer_out.append(individual_layer_out)

    # Concatenate the output of the individual convolutional layers then apply dropout.
    sensor_conv_in = tf.concat(sensors_indiv_layer_out, 2)
    # sensor_conv_in has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, num_sensors, ...(depends on kernel_size and padding of conv layers), CHANNELS).
    senor_conv_shape_l = sensor_conv_in.get_shape().as_list()
    senor_conv_shape = tf.shape(sensor_conv_in)
    sensor_conv_in = tf.layers.dropout(sensor_conv_in, rate=0.2,
        noise_shape=[senor_conv_shape[0], 1, 1, 1, senor_conv_shape[4]], training=(mode == tf.estimator.ModeKeys.TRAIN))
    sensor_conv_in.set_shape([None, senor_conv_shape_l[1], num_sensors, senor_conv_shape_l[3], 64])
    
    #------ Self-Attention Module
    merge_sensors = sa_conv_module(sensor_conv_in, num_sensors, number_of_outputs)
    
    #------ Merge Convolutional Layers    
    merge_conv1 = tf.layers.conv2d(merge_sensors, filters=64, kernel_size=[1, 8], strides=[1, 1], padding="same")
    merge_conv1 = tf.layers.batch_normalization(merge_conv1, training=(mode == tf.estimator.ModeKeys.TRAIN))
    merge_conv1 = tf.nn.relu(merge_conv1)
    merge_conv1_shape = tf.shape(merge_conv1)
    merge_conv1 = tf.layers.dropout(merge_conv1, rate=0.2,
        noise_shape=[merge_conv1_shape[0], 1, 1, merge_conv1_shape[3]], training=(mode == tf.estimator.ModeKeys.TRAIN))
        
    merge_conv2 = tf.layers.conv2d(merge_conv1, filters=64, kernel_size=[1, 6], strides=[1, 1], padding="same")
    merge_conv2 = tf.layers.batch_normalization(merge_conv2, training=(mode == tf.estimator.ModeKeys.TRAIN))
    merge_conv2 = tf.nn.relu(merge_conv2)
    merge_conv2_shape = tf.shape(merge_conv2)
    merge_conv2 = tf.layers.dropout(merge_conv2, rate=0.2,
        noise_shape=[merge_conv2_shape[0], 1, 1, merge_conv2_shape[3]], training=(mode == tf.estimator.ModeKeys.TRAIN))
        
    merge_conv3 = tf.layers.conv2d(merge_conv2, filters=64, kernel_size=[1, 4], strides=[1, 1], padding="same")
    merge_conv3 = tf.layers.batch_normalization(merge_conv3, training=(mode == tf.estimator.ModeKeys.TRAIN))
    merge_conv3 = tf.nn.relu(merge_conv3)
    
    return merge_conv3    
    

def stacked_gru_layer(layer_input, sequence_lengths, number_of_cells, mode):
    batch_size = tf.shape(layer_input)[0]
    gru_cell1 = tf.contrib.rnn.GRUCell(number_of_cells)
    if mode == tf.estimator.ModeKeys.TRAIN:
        gru_cell1 = tf.contrib.rnn.DropoutWrapper(gru_cell1, output_keep_prob=0.5)

    gru_cell2 = tf.contrib.rnn.GRUCell(number_of_cells)
    if mode == tf.estimator.ModeKeys.TRAIN:
        gru_cell2 = tf.contrib.rnn.DropoutWrapper(gru_cell2, output_keep_prob=0.5)

    cell = tf.contrib.rnn.MultiRNNCell([gru_cell1, gru_cell2])
    init_state = cell.zero_state(batch_size, tf.float32)
    
    cell_output, final_stateTuple = tf.nn.dynamic_rnn(cell, layer_input, sequence_length=sequence_lengths, initial_state=init_state, time_major=False)
    # cell_output has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, number_of_cells).
    
    return cell_output
    

def bidirectional_gru_layer(layer_input, sequence_lengths, number_of_cells, mode):
    batch_size = tf.shape(layer_input)[0]
    gru_cell_fw = tf.contrib.rnn.GRUCell(number_of_cells)
    if mode == tf.estimator.ModeKeys.TRAIN:
        gru_cell_fw = tf.contrib.rnn.DropoutWrapper(gru_cell_fw, output_keep_prob=0.5)

    gru_cell_bw = tf.contrib.rnn.GRUCell(number_of_cells)
    if mode == tf.estimator.ModeKeys.TRAIN:
        gru_cell_bw = tf.contrib.rnn.DropoutWrapper(gru_cell_bw, output_keep_prob=0.5)

    init_state_fw = gru_cell_fw.zero_state(batch_size, tf.float32)
    init_state_bw = gru_cell_bw.zero_state(batch_size, tf.float32)

    cell_outputs, final_stateTuple = tf.nn.bidirectional_dynamic_rnn(gru_cell_fw, gru_cell_bw, layer_input, sequence_length=sequence_lengths, 
                                                                    initial_state_fw=init_state_fw, initial_state_bw=init_state_bw, time_major=False)
    # cell_output is a tuple (output_fw, output_bw) where each output has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, number_of_cells).
    # we concatenate the two and pass them through a dense layer to combine them to an output of size (BATCH_SIZE, NUMBER_OF_INTERVALS, 120)
    cell_output = tf.concat(cell_outputs, 2)
    return tf.layers.dense(cell_output, 120)
    

def attention_gru(layer_input, sequence_lengths, number_of_cells=120):
    layer_input_shape = layer_input.get_shape().as_list()
    batch_size = tf.shape(layer_input)[0]
    num_intervals = 20
    num_channels = layer_input_shape[2]
    features_dim = layer_input_shape[3]
    
    # Get initial GRU cells state, from first timestep data
    batch_at_first_timestep = tf.slice(layer_input, [0, 0, 0, 0], [batch_size, 1, num_channels, features_dim])
    batch_at_first_timestep = tf.reshape(batch_at_first_timestep, [batch_size, num_channels, features_dim])
    feature_channel_mean = tf.reduce_mean(batch_at_first_timestep, axis=2)
    # feature_channel_mean has shape (BATCH_SIZE, num_channels)
    w_h = tf.get_variable('w_h', [num_channels, number_of_cells])
    b_h = tf.get_variable('b_h', [number_of_cells])
    initial_rnn_state = tf.nn.tanh(tf.matmul(feature_channel_mean, w_h) + b_h)
    
    gru_cell = tf.contrib.rnn.GRUCell(number_of_cells)
    state = initial_rnn_state
    outputs = []
    # we use a mask to take different sequence lengths into account (zero out what's outside)
    length_masks = tf.sequence_mask(sequence_lengths, maxlen=num_intervals, dtype=tf.float32) # shape (batch_size, num_intervals)
    
    # Define learnable matrices to calculate attention scores
    w_att_feat = tf.get_variable('w_att_feat', [num_channels*features_dim, num_channels])
    b_att_feat = tf.get_variable('b_att_feat', [num_channels])
    w_att_hidd = tf.get_variable('w_att_hidd', [number_of_cells, num_channels])
    b_att_hidd = tf.get_variable('b_att_hidd', [num_channels])
    
    for i in range(num_intervals):
        current_timestep_data = tf.slice(layer_input, [0, i, 0, 0], [batch_size, 1, num_channels, features_dim])
        current_timestep_data = tf.reshape(current_timestep_data, [batch_size, num_channels, features_dim])
        
        # Attention score
        current_timestep_data_flattened = tf.reshape(current_timestep_data, [batch_size, num_channels*features_dim])
        score_feat = tf.matmul(current_timestep_data_flattened, w_att_feat) + b_att_feat
        score_hidd = tf.matmul(state, w_att_hidd) + b_att_hidd 
        score = tf.nn.tanh(score_feat + score_hidd)
        # score has shape (batch_size, num_channels)
        
        # Attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        attention_weights = tf.reshape(attention_weights, [batch_size, num_channels, 1])
        
        # Context vector
        context_vector = current_timestep_data * attention_weights 
        context_vector = tf.reduce_sum(context_vector, axis=1)
        
        # GRU
        input = tf.concat([current_timestep_data_flattened, context_vector], axis=-1)
        output, state = gru_cell(input, state)
        
        # take length into account (zero out what's outside of sequence length)
        timestamp_mask = tf.slice(length_masks, [0, i], [batch_size, 1])
        output = output * timestamp_mask    
        output = tf.reshape(output, [batch_size, 1, number_of_cells])
        # output has shape (batch_size, 1, feature_dim)
        outputs.append(output)
        
    cell_outputs = tf.concat(outputs, axis=1)
    # cell_output has shape (batch_size, num_intervals, number_of_cells)
    return cell_outputs


def multi_head_attention(input, number_of_heads=8):
        # input has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, sensor_conv_out_shape[2])
        input_shape = input.get_shape().as_list()
        
        # multi heads
        z_array = []
        d_k_val = 64
        d_k = tf.constant(d_k_val, tf.float32) # dimension of query, key, value vectors
        for i in range(number_of_heads):
            w_q = tf.get_variable("w_q_"+str(i), shape=(1, input_shape[2], d_k_val))
            w_k = tf.get_variable("w_k_"+str(i), shape=(1, input_shape[2], d_k_val))
            w_v = tf.get_variable("w_v_"+str(i), shape=(1, input_shape[2], d_k_val))
            q = tf.einsum('bnk,ikm->bnm', input, w_q)
            k = tf.einsum('bnk,ikm->bnm', input, w_k)
            v = tf.einsum('bnk,ikm->bnm', input, w_v)
            # q, k and v have shape (BATCH_SIZE, NUMBER_OF_INTERVALS, d_k)

            score = tf.matmul(q, tf.transpose(k, perm=[0, 2, 1])) / tf.sqrt(d_k)
            # score has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, NUMBER_OF_INTERVALS)
            weights = tf.nn.softmax(score, axis=-1)

            z = tf.matmul(weights, v)
            # z has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, d_k)
            z_array.append(z)

        z = tf.concat(z_array, axis=-1)
        # z has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, d_k * number_of_heads)
        w_0 = tf.get_variable("w_0", shape=(1, d_k_val*number_of_heads, input_shape[2]))
        z_final = tf.einsum('bnk,ikm->bnm', z, w_0)
        #z final has shape (batch_size, NUMBER_OF_INTERVALS, input_shape[2])
        return z_final


def transformer_encoder(layer_input):
    #####
    #layer_input = tf.layers.dense(layer_input, units=120, activation=tf.nn.relu) 

    input_shape = layer_input.get_shape().as_list()
    batch_size = tf.shape(layer_input)[0]
    number_of_intervals = 20

    attention_heads_output = multi_head_attention(layer_input)

    # Add & Normalize
    sum_1 = layer_input + attention_heads_output
    normalized_1 = tf.contrib.layers.layer_norm(sum_1)

    # Feed-forward
    """ff_in = tf.reshape(normalized_1, (batch_size, number_of_intervals * input_shape[2]))
    ff_out_intermediate = tf.layers.dense(ff_in, units=number_of_intervals * input_shape[2], activation=tf.nn.relu) 
    ff_out = tf.layers.dense(ff_out_intermediate, units=number_of_intervals * input_shape[2])
    ff_out = tf.reshape(ff_out, (batch_size, number_of_intervals, -1))"""
    ff_out_intermediate = tf.layers.conv1d(normalized_1, input_shape[2], 1)
    ff_out = tf.layers.conv1d(ff_out_intermediate, input_shape[2], 1)

    # Add & Normalize
    sum_2 = normalized_1 + ff_out
    normalized_2 = tf.contrib.layers.layer_norm(sum_2)
    # normalized has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, layer_input[2])
    return normalized_2

    
def output_layer(layer_input, number_of_output_classes):
    # The layer will be name last so it's weights can be easily extracted (for transfer learning)
    # once the network is trained.
    logits = tf.layers.dense(layer_input, number_of_output_classes, name="last")
    return logits