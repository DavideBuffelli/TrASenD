import os
from shutil import copy


def create_folder_structure(number_of_subjects, output_folder):
    for i in range(1, number_of_subjects+1):
        subject_folder = os.path.join(output_folder, "Subject_"+str(i))
        os.mkdir(subject_folder)
        os.mkdir(os.path.join(subject_folder, "train"))
        os.mkdir(os.path.join(subject_folder, "eval"))
        
def distribute_training_subject_sample(subject_folder, number_of_subjects, output_folder):
    for activity in os.listdir(subject_folder):
        if activity.startswith("."):
            continue
        activity_folder = os.path.join(subject_folder, activity)
        for sample in os.listdir(activity_folder):
            for i in range(1, number_of_subjects+1):
                if "Subject_"+str(i) not in subject_folder:
                    subject_train_folder = os.path.join(output_folder, "Subject_"+str(i), "train")
                    copy(os.path.join(activity_folder, sample), subject_train_folder)
                    
def prepare_eval_folder(subject_folder, subject_number, output_folder):
    for activity in os.listdir(subject_folder):
        if activity.startswith("."):
            continue
        activity_folder = os.path.join(subject_folder, activity)
        sorted_samples = sorted(os.listdir(activity_folder), key=lambda x: int(x.split(".")[0]))
        for i in range(0, len(sorted_samples), 10): # don't take the augmentations
            sample = sorted_samples[i]
            subject_eval_folder = os.path.join(output_folder, "Subject_"+str(subject_number), "eval")
            copy(os.path.join(activity_folder, sample), subject_eval_folder)


if __name__ == "__main__":
    dataset_folder = "../USC-HAD/Preprocessed"
    output_folder = "../USC-HAD/Ready"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    number_of_subjects = len([x for x in os.listdir(dataset_folder) if not x.startswith(".")])
    create_folder_structure(number_of_subjects, output_folder)
    
    for subject in os.listdir(dataset_folder):
        if subject.startswith("."):
            continue
        subject_folder = os.path.join(dataset_folder, subject)
        distribute_training_subject_sample(subject_folder, number_of_subjects, output_folder)
        prepare_eval_folder(subject_folder, int(subject.split("_")[1]), output_folder)