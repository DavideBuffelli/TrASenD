import csv
import numpy as np
import os


pamap2_dir = "../datasets_HAR/balanced_PAMAP2_2"

for user_folder in os.listdir(pamap2_dir):
    user_folder_path = os.path.join(pamap2_dir, user_folder)
    #print("user_folder_path", user_folder_path)
    for train_eval_folder in os.listdir(user_folder_path):
        train_eval_folder_path = os.path.join(user_folder_path, train_eval_folder)
        #print("train", train_eval_folder_path)
        for file in os.listdir(train_eval_folder_path):
            file_path = os.path.join(train_eval_folder_path, file)
            print(file_path)
            with open(file_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                data = next(csv_reader)
            
            features = data[:-24]
            labels = np.array([float(x) for x in data[-24:]])
            good_indexes = [0, 1, 2, 3, 4, 5, 6, 11, 12, 15, 16, 23]
            good_labels = labels[good_indexes].tolist()
            
            with open(file_path, "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=',')
                writer.writerow(features+[str(x) for x in good_labels])
