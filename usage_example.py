import tensorflow as tf

from input_utils import input_fn, predict_input_fn
from models import LSTM, DeepSense, SADeepSense, TrASenD_BD, TrASenD_CA, TrASenD


if __name__ == "__main__":
    # Directory Paths
    TRAINING_DATA_FOLDER_PATH = "/..path../sepHARData_a/train"
    EVAL_DATA_FOLDER_PATH = "/..path../sepHARData_a/eval"
    MODEL_DIR_PATH = "/path/to/model/dir"
    
    # -------------- GET ESTIMATOR
    # Wrap DeepSense estimator in a tf.estimator.Estimator, passing all parameters.
    # These are the ones specified by the authors of the DeepSense framework.
    num_sensors = 2
    batch_size = 64
    hyper_params = {"learning_rate": 1e-4,
                    "beta1": 0.5,
                    "beta2": 0.9, 
                    "l2_lambda_term": 5e-4}
    num_output_classes = 6
    
    ds_model = DeepSense(num_sensors, num_output_classes, hyper_params)
                    
    deepSense_classifier = tf.estimator.Estimator(
        model_fn = ds_model.get_model_function(),
        model_dir = MODEL_DIR_PATH,
        params = None)

    # -------------- TRAIN ESTIMATOR
    # tf.estimator.Estimator wants an input function with no arguments, so wrap input_fn in a lambda.
    training_input_function = lambda: input_fn(TRAINING_DATA_FOLDER_PATH, batch_size, num_sensors, num_output_classes, True)
    deepSense_classifier.train(training_input_function, steps=1)

    # -------------- EVALUATE METRICS
    eval_input_function = lambda: input_fn(EVAL_DATA_FOLDER_PATH, batch_size, num_sensors, num_output_classes, False)
    eval_result = deepSense_classifier.evaluate(eval_input_function, steps=10)
    print("\nTest Set Accuracy: {accuracy:0.3f}\nMean per Class Accuracy: {mean_perClass_accuracy:0.3f}".format(**eval_result))

    # -------------- EXAMPLE OF PREDICTION
    predict_input_function = lambda: predict_input_fn("/Users/davidebuffelli/Desktop/final/sepHARData_a/eval/eval_0.csv", num_sensors, num_output_classes)
    predictions = deepSense_classifier.predict(predict_input_function)
    for p in predictions:
        print("Predicted Class: ", p["class_ids"])
        print("Probabilities: ", p["probabilities"])
        print("Logits: ", p["logits"])
        print("Transfer Learning input: ", p["tl_input"])