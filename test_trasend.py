import numpy as np
import os
import shutil
import tensorflow as tf

from input_utils import input_fn, predict_input_fn
from models import TrASenD


HHAR_DATA_FOLDER_PATH = "/path/to/hhar/data/dir"
MODEL_DIR_PATH = "/path/to/model/dir"

if not os.path.exists(MODEL_DIR_PATH):
    os.mkdir(MODEL_DIR_PATH)

# Model Parameters
num_sensors = 2
batch_size = 64
hyper_params = {"learning_rate": 1e-3,
                "beta1": 0.5,
                "beta2": 0.9, 
                "l2_lambda_term": 5e-4}
num_output_classes = 6

trasend_model = TrASenD(num_sensors, num_output_classes, hyper_params)
        
f1_scores = []
for user_data_folder in os.listdir(HHAR_DATA_FOLDER_PATH):
    if user_data_folder.startswith("."):
        continue
    full_user_data_folder = os.path.join(HHAR_DATA_FOLDER_PATH, user_data_folder)
    current_user = user_data_folder.split("_")[1]
    print("----- Training and evaluating for User:", current_user)
    training_data_folder = os.path.join(full_user_data_folder, "train")
    eval_data_folder = os.path.join(full_user_data_folder, "eval")
    current_user_model_dir = os.path.join(MODEL_DIR_PATH, "user_{}_model_dir".format(current_user))
    if os.path.exists(current_user_model_dir):
        shutil.rmtree(current_user_model_dir)
    os.mkdir(current_user_model_dir)
    
    # Create TensorFlow estimator
    trasend_estimator = tf.estimator.Estimator(
                            model_fn = trasend_model.get_model_function(),
                            model_dir = current_user_model_dir,
                            params = None)
    
    # Train & Evaluate
    best_f1_score = 0
    for epoch in range(30):
        # Train
        training_input_function = lambda: input_fn(training_data_folder, 
                                                   batch_size, 
                                                   num_sensors, 
                                                   num_output_classes, 
                                                   True)
        trasend_estimator.train(training_input_function)
    
        # Eval
        eval_input_function = lambda: input_fn(eval_data_folder, 
                                               batch_size, 
                                               num_sensors, 
                                               num_output_classes, 
                                               False)
        eval_result = trasend_estimator.evaluate(eval_input_function)
        cm = eval_result["conf_matrix"]
        # Calculate F1-score from confusion matrix
        TP = np.diag(cm)
        FP = np.sum(cm, axis=0) - TP
        FN = np.sum(cm, axis=1) - TP
        num_classes = cm.shape[0]
        TN = []
        for i in range(num_classes):
            temp = np.delete(cm, i, 0)    # delete ith row
            temp = np.delete(temp, i, 1)  # delete ith column
            TN.append(sum(sum(temp)))    
        precision = TP/((TP+FP)+0.01)
        recall = TP/((TP+FN)+0.01)
        precision = np.clip(precision, 0, 1)
        recall = np.clip(recall, 0, 1)
        for i in range(TP.shape[0]):
            if TP[i] == 0 and FP[i] == 0 and FN[i] == 0:
                precision[i] = 1
                recall[i] = 1
            if TP[i] == 0 and (FP[i] > 0 or FN[i] > 0):
                precision[i] = 0
                recall[i] = 0
        f1_score = 2 * ( (precision * recall) / (precision + recall) )
        f1_score = np.nanmean(f1_score)
        print("Epoch: {}, F1-score: {}".format(epoch, f1_score))
        if f1_score > best_f1_score:
            best_f1_score = f1_score
    print("Best F1-score for user {}: {}".format(current_user, best_f1_score))
    f1_scores.append(best_f1_score)
    
f1_scores = np.array(f1_scores)
print("Cross Validation F1-score:", f1_scores.mean())