import tensorflow as tf


''' Constants that are used for all datasets. '''
class Constants:
    SAMPLE_LENGTH = 5.0 # Length in seconds of each sample that is contained in a .csv file.
    TAO = 0.25 # Interval length
    NUMBER_OF_INTERVALS = int(SAMPLE_LENGTH / TAO)
    MEASUREMENTS_PER_INTERVAL = 10 # what is called f in the paper
    MEASUREMENTS_DIMENSIONS = 3 # dimension of the measurements coming from the sensors
    
    @staticmethod
    def get_num_features_per_interval(number_of_sensors):
        return number_of_sensors * (Constants.MEASUREMENTS_DIMENSIONS * 2) * Constants.MEASUREMENTS_PER_INTERVAL
        
    @staticmethod
    def get_num_features_per_sample(number_of_sensors):
        return Constants.NUMBER_OF_INTERVALS * Constants.get_num_features_per_interval(number_of_sensors)


def prepare_features(features, number_of_intervals, number_of_features, mode):
    # When we are in TRAIN or in EVAL mode, features has shape (BATCH_SIZE, NUMBER_OF_INTERVALS, FEATURE_DIM)
    # but, when we are in PREDICT mode features corresponds to a single element, so it has shape
    # (NUMBER_OF_INTERVALS, FEATURE_DIM). 
    # TensorFlow methods require an input of shape (BATCH_SIZE, NUMBER_OF_INTERVALS, FEATURE_DIM, CHANNELS)
    # se we have to do some reshaping.
    f = features["features"]
    length = features["length"]
    if mode == tf.estimator.ModeKeys.PREDICT:
        f = tf.reshape(f, [1, number_of_intervals, number_of_features])
        length = tf.reshape(length, [1]) # Make it a tensor instead of a scalar.
    f = tf.expand_dims(f, axis=-1) # Add dimension for the channel.
    
    return f, length


def cross_entropy_loss(model_output, true_labels):
    batchLoss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=model_output, labels=true_labels)
    loss = tf.reduce_mean(batchLoss)
    return loss
    

def l2_regularization_loss(lambda_term=1e-2):
    vars = tf.trainable_variables()
    lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if "bias" not in v.name]) * lambda_term
    return lossL2
    

def eval_confusion_matrix(labels, predictions, num_output_classes):
    with tf.variable_scope("eval_confusion_matrix"):
        con_matrix = tf.confusion_matrix(labels=labels, predictions=predictions, num_classes=num_output_classes)

        con_matrix_sum = tf.Variable(tf.zeros(shape=(num_output_classes, num_output_classes), dtype=tf.int32), trainable=False,
            name="confusion_matrix_result",
            collections=[tf.GraphKeys.LOCAL_VARIABLES])

        update_op = tf.assign_add(con_matrix_sum, con_matrix)

        return tf.convert_to_tensor(con_matrix_sum), update_op
        

def get_eval_estimatorspec(model_output, true_labels, loss, num_output_classes):
    model_predictions = tf.argmax(model_output, 1)
    labels = tf.argmax(true_labels, 1)
    
    accuracy = tf.metrics.accuracy(labels=labels, predictions=model_predictions, name="accuracy_op")
    mean_perClass_accuracy = tf.metrics.mean_per_class_accuracy(labels, model_predictions, num_output_classes, name="mean_perClass_accuracy_op")
    conf_matrix = eval_confusion_matrix(labels, model_predictions, num_output_classes)
    metrics = {"accuracy": accuracy, "mean_perClass_accuracy": mean_perClass_accuracy, "conf_matrix": conf_matrix}
    
    tf.summary.scalar("Accuracy", accuracy[1]) # for TensorBoard.
    tf.summary.scalar("Mean Per Class Accuracy", mean_perClass_accuracy[1]) # for TensorBoard.
    return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=metrics)
    
    
def get_predict_estimatorspec(model_output, tl_input):
    predicted_classes = tf.argmax(model_output, 1)

    predictions = {"class_ids": predicted_classes[:, tf.newaxis],
                   "probabilities": tf.nn.softmax(model_output),
                   "logits": model_output,
                   "tl_input": tl_input}
    # tl_input is needed only for transfer learning(it becomes the input of the user-specific output layer).

    return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.PREDICT, predictions=predictions, 
        export_outputs={"classify:":tf.estimator.export.PredictOutput(predictions)})
        
        
def get_training_estimatorspec(loss, hyer_params):
    optimizer = tf.train.AdamOptimizer(
        learning_rate=hyer_params["learning_rate"],
        beta1=hyer_params["beta1"],
        beta2=hyer_params["beta2"])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
    return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)