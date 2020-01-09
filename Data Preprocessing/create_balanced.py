import csv
import numpy as np
import os
import random
from shutil import copy


#users = ["a", "b", "c", "d", "e", "f", "g", "h", "i"] # HHAR
users = [i for i in range(1, 13)] # USCHAD
#users = [i for i in range(1, 10)] # PAMAP2
#min_count = {"a": 7440, "b": 7610, "c": 6530, "d": 8120, "e": 7700, "f": 8090, "g": 6790, "h": 7260, "i": 8570} # HHAR
min_count = {"1": 123, "2": 225, "3": 223, "4": 226, "5": 230, "6": 230, "7": 223, "8": 226, "9": 222, "10": 231, "11": 219, "12": 212} # USCHAD_no_aug
#min_count = {"1": 1230, "2": 2250, "3": 2230, "4": 2260, "5": 2300, "6": 2300, "7": 2230, "8": 2260, "9": 2220, "10": 2310, "11": 2190, "12": 2120} # USCHAD
#min_count = {"1": 750, "2": 740, "3": 1010, "4": 1010, "5": 850, "6": 1000, "7": 1010, "8": 830, "9": 880} # PAMAP2
#num_activities = 6 # HHAR
num_activities = 12 # USCHAD
#num_activities = 24 # PAMAP2

#users_folders_path = "/nfsd/VFdisk/buffelli/datasets_HAR/HHAR/"
users_folders_path = "/nfsd/VFdisk/buffelli/datasets_HAR/USC_HAD_no_aug"
output_dir = "/nfsd/VFdisk/buffelli/datasets_HAR/balanced_USCHAD_no_aug/"
if not os.path.exists(output_dir):
	os.mkdir(output_dir)

for user in users:
	user = str(user)
	print("User:", user)
	print("min:count", min_count[user])
	#train_dir = os.path.join(users_folders_path, "sepHARData_"+user, "train")
	train_dir = os.path.join(users_folders_path, "Subject_"+user, "train")	

	files_per_activity = dict()
	
	#user_out_dir = os.path.join(output_dir, "sepHARData_"+user)
	user_out_dir = os.path.join(output_dir, "Subject_"+user)
	os.mkdir(user_out_dir)
	user_train_out_dir = os.path.join(user_out_dir, "train")
	os.mkdir(user_train_out_dir)
		
	for train_file in os.listdir(train_dir):
		file_path = os.path.join(train_dir, train_file)
		with open(file_path, "r") as csvfile:
			csv_reader = csv.reader(csvfile)
			line = next(csv_reader)
			labels = "".join(line[-num_activities:])
			
			if labels in files_per_activity:
				files_per_activity[labels].append(file_path)
			else:
				files_per_activity[labels] = [file_path]
				
	print("Keys:", files_per_activity.keys())
				
	for label, files in files_per_activity.items():
		print("Label:", label)
		print("files count:", len(files))
		files_to_be_copied = random.sample(files, min_count[user])
		for file in files_to_be_copied:
			copy(file, user_train_out_dir)
			
	#eval_dir = os.path.join(users_folders_path, "sepHARData_"+user, "eval")
	eval_dir = os.path.join(users_folders_path, "Subject_"+user, "eval")
	user_eval_out_dir = os.path.join(user_out_dir, "eval")
	os.mkdir(user_eval_out_dir)
		
	for file in os.listdir(eval_dir):
		file_path = os.path.join(eval_dir, file)
		copy(file_path, user_eval_out_dir)
		
				
		 
