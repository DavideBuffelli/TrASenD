import csv
import numpy as np
import os
import scipy.io
from scipy.interpolate import interp1d
from scipy.fftpack import fft


def extract_data(filename):
    # Output dataformat we want:
    # timestamp, activityid, imu_hand, imu_chest, imu_ankle
    # with imu_* containing: x_acc, y_acc, z_acc, x_gyro, y_gyro, z_gyro, x_mag, y_mag, z_mag,
    columns_of_interest = [0, 1, 4, 5, 6, 10, 11, 12, 13, 14, 15, 21, 
                           22, 23, 27, 28, 29, 30, 31, 32, 38, 39, 40,
                           44, 45, 46, 47, 48, 49]

    data = []
    with open(filename, "r") as file:
        for line in file:
            columns = line.split(" ")
            if columns[1] == "0": # transient activity recordings
                continue
            interesting_columns = (np.array(columns)[columns_of_interest]).astype(np.float)
            data.append(interesting_columns)
    
    return np.vstack(data)

def divide_by_activity(data):
    data = data[data[:, 1].argsort()] # sort by activity index
    
    activity_starting_indexes = []
    current_activity = None
    for i in range(0, len(data)):
        if data[i, 1] != current_activity:
            current_activity = data[i, 1]
            activity_starting_indexes.append(i)
    return activity_starting_indexes

def get_five_seconds_samples(readings):
    ordered_readings = readings[readings[:, 0].argsort()] # sort by timestamp
    
    start_time = ordered_readings[0, 0]
    end_time = start_time + 5
    samples = []
    current_sample = []
    for reading in ordered_readings:
        if reading[0] < end_time:
            current_sample.append(reading)
        else:
            if len(current_sample) > 200:
                samples.append(np.vstack(current_sample))
            current_sample = []
            start_time = reading[0]
            end_time = start_time + 5
            current_sample.append(reading)
    if len(current_sample) > 200: # otherwise it's just too small
        samples.append(np.vstack(current_sample))
        
    return samples

def add_noise(sensor_readings, acc_noise_var, gyro_noise_var, mag_noise_var):
    num_readings = len(sensor_readings)
    acc_noise_arrays_1 = [np.random.normal(0.0, acc_noise_var, size=(3,)) for _ in range(num_readings)]
    acc_noise_matrix_1 = np.vstack(acc_noise_arrays_1)
    acc_noise_arrays_2 = [np.random.normal(0.0, acc_noise_var, size=(3,)) for _ in range(num_readings)]
    acc_noise_matrix_2 = np.vstack(acc_noise_arrays_2)
    acc_noise_arrays_3 = [np.random.normal(0.0, acc_noise_var, size=(3,)) for _ in range(num_readings)]
    acc_noise_matrix_3 = np.vstack(acc_noise_arrays_3)
    gyro_noise_arrays_1 = [np.random.normal(0.0, gyro_noise_var, size=(3,)) for _ in range(num_readings)]
    gyro_noise_matrix_1 = np.vstack(gyro_noise_arrays_1)
    gyro_noise_arrays_2 = [np.random.normal(0.0, gyro_noise_var, size=(3,)) for _ in range(num_readings)]
    gyro_noise_matrix_2 = np.vstack(gyro_noise_arrays_2)
    gyro_noise_arrays_3 = [np.random.normal(0.0, gyro_noise_var, size=(3,)) for _ in range(num_readings)]
    gyro_noise_matrix_3 = np.vstack(gyro_noise_arrays_3)
    mag_noise_arrays_1 = [np.random.normal(0.0, mag_noise_var, size=(3,)) for _ in range(num_readings)]
    mag_noise_matrix_1 = np.vstack(mag_noise_arrays_1)
    mag_noise_arrays_2 = [np.random.normal(0.0, mag_noise_var, size=(3,)) for _ in range(num_readings)]
    mag_noise_matrix_2 = np.vstack(mag_noise_arrays_2)
    mag_noise_arrays_3 = [np.random.normal(0.0, mag_noise_var, size=(3,)) for _ in range(num_readings)]
    mag_noise_matrix_3 = np.vstack(mag_noise_arrays_3)
    comb_noise_matrix = np.concatenate((np.zeros((num_readings, 2)), acc_noise_matrix_1, gyro_noise_matrix_1, mag_noise_matrix_1,
                                                                     acc_noise_matrix_2, gyro_noise_matrix_2, mag_noise_matrix_2,
                                                                     acc_noise_matrix_3, gyro_noise_matrix_3, mag_noise_matrix_3), axis=1)
    sensor_readings += comb_noise_matrix
    return sensor_readings
    
def augmented_readings_generator(sensor_readings, num_augmentations, acc_noise_var=0.5, gyro_noise_var=0.2, mag_noise_var=0.2):
    yield sensor_readings
    for _ in range(num_augmentations-1):
        sr_copy = sensor_readings.copy()
        yield add_noise(sr_copy, acc_noise_var, gyro_noise_var, mag_noise_var)
        
def divide_in_sub_intervals(sensor_readings, sub_interval_length):
    sub_intervals = []
    for i in range(0, len(sensor_readings), sub_interval_length):
        if len(sensor_readings[i:i+sub_interval_length]) > 1:
            sub_intervals.append(sensor_readings[i:i+sub_interval_length, :])
    return sub_intervals

def interp_and_fft(measurements):
    interval_times = np.arange(0, len(measurements))
    interpolation_function = interp1d(interval_times, measurements, axis=0, assume_sorted=True)
    points = np.linspace(interval_times[0], interval_times[-1], 10)
    interp_values = interpolation_function(points)
    FFT_values = fft(interp_values)
    return FFT_values # returns a list of complex-valued arrays.
    
def append_fft_values_to_output(output, fft_values):
    for fft_value in fft_values:
        for elem in fft_value:
            output.append(elem.real)
            output.append(elem.imag)
            
def get_sub_interval_fft(sub_interval):
    measurements = sub_interval[:, 2:]
    out = []
    for i in range(0, measurements.shape[1]-1, 3):
        current_sensor_measurements = measurements[:, i:i+3]
        fft_values = interp_and_fft(current_sensor_measurements)
        out.append(fft_values)
    return out # out is a list with 9 np.array (one per sensor) of complex values with shape (10, 3) 
    
def preprocess_and_get_output_list(sub_intervals):
    output = []
    for sub_interval in sub_intervals:
        sensors_fft_values = get_sub_interval_fft(sub_interval)
        for sensor_values in sensors_fft_values:
            append_fft_values_to_output(output, sensor_values)
    return output
    
def pad_output_list_to_length(output_list, length):
    if len(output_list) < length:
        output_list.extend([0.0 for _ in range(length-len(output_list))])
        
def append_activity_encoding(output, activity):
    num_activities = 24
    gt_encoding = {i: [1.0 if j == i-1 else 0.0 for j in range(num_activities)] for i in range(1, num_activities+1)}
    output.extend(gt_encoding[activity])

def create_output_folders(dataset_output_folder, num_subjects, num_activities):
    for s in range(1, num_subjects+1):
        subject_folder = os.path.join(dataset_output_folder, "Subject_"+str(s))
        os.mkdir(subject_folder)
        for a in range(1, num_activities+1):
            os.mkdir(os.path.join(subject_folder, "Activity_"+str(a)))

def write_output_to_csv(output, dataset_output_folder, subject, activity, index):
    output_file_name = os.path.join(dataset_output_folder, "Subject_"+str(subject), "Activity_"+str(int(activity)), str(index)+".csv")
    with open(output_file_name, "w+") as writeFile:
        writer = csv.writer(writeFile)
        writer.writerow(output)
            
if __name__ == "__main__":
    dataset_output_folder = "../Data/PAMAP2_Dataset/Preprocessed"
    sample_index = 0
    create_output_folders(dataset_output_folder, 9, 24)
    
    dataset_folder = "../Data/PAMAP2_Dataset/Protocol/"
    for subject_file in os.listdir(dataset_folder):
        print("Processing file:", subject_file)
        full_file_path = os.path.join(dataset_folder, subject_file)
        og_data = extract_data(full_file_path)
        activity_indexes = divide_by_activity(og_data)
        
        start_idx = activity_indexes.pop(0)
        len_data = len(og_data)
        activity_indexes.append(len_data)
        for data in augmented_readings_generator(og_data, 10):
            for idx in activity_indexes:
                activity_sensor_readings = data[start_idx:idx]
                for sample in get_five_seconds_samples(activity_sensor_readings):
                    sub_intervals = divide_in_sub_intervals(sample, 25)
                    output = preprocess_and_get_output_list(sub_intervals)
                    pad_output_list_to_length(output, 10800)
                    append_activity_encoding(output, sample[0, 1])
                    if len(output) != 10824:
                        print("Error")
                    write_output_to_csv(output, dataset_output_folder, int(subject_file[-5]), sample[0, 1], sample_index)
                    sample_index += 1
                    
                start_idx = idx
                if start_idx == len_data:
                    break
            start_idx = 0
                    
        print(sample_index)