import csv
import numpy as np
import os
import scipy.io
from scipy.interpolate import interp1d
from scipy.fftpack import fft


def get_subject_index(subject_folder_name):
    return int(subject_folder_name.split("t")[1])

def get_activity_index(readings_file_name):
    activity_string = readings_file_name.split("t")[0]
    return int(activity_string[1:])

def get_sensor_reading_from_file(file): 
    mat_file = scipy.io.loadmat(file)
    return mat_file["sensor_readings"]
    
def get_five_seconds_samples(sensor_readings):
    return divide_in_sub_intervals(sensor_readings, 500)
    
def add_noise(sensor_readings, acc_noise_var, gyro_noise_var):
    num_readings = len(sensor_readings)
    acc_noise_arrays = [np.random.normal(0.0, acc_noise_var, size=(3,)) for _ in range(num_readings)]
    acc_noise_matrix = np.vstack(acc_noise_arrays)
    gyro_noise_arrays = [np.random.normal(0.0, gyro_noise_var, size=(3,)) for _ in range(num_readings)]
    gyro_noise_matrix = np.vstack(gyro_noise_arrays)
    comb_noise_matrix = np.concatenate((acc_noise_matrix, gyro_noise_matrix), axis=1)
    sensor_readings += comb_noise_matrix
    return sensor_readings
    
def augmented_readings_generator(sensor_readings, num_augmentations, acc_noise_var=0.5, gyro_noise_var=0.2):
    yield sensor_readings
    for _ in range(num_augmentations-1):
        sr_copy = sensor_readings.copy()
        yield add_noise(sr_copy, acc_noise_var, gyro_noise_var)

def divide_in_sub_intervals(sensor_readings, sub_interval_length):
    sub_intervals = []
    for i in range(0, len(sensor_readings), sub_interval_length):
        if len(sensor_readings[i:i+sub_interval_length, :]) > 1:
            sub_intervals.append(sensor_readings[i:i+sub_interval_length, :])
    return sub_intervals
    
def interp_and_fft(measurements):
    interval_times = np.arange(0, len(measurements))
    interpolation_function = interp1d(interval_times, measurements, axis = 0)
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
    acc_values = sub_interval[:, :3]
    gyro_values = sub_interval[:, 3:]
    acc_FFT_values = interp_and_fft(acc_values)
    gyro_FFT_values = interp_and_fft(gyro_values)
    return acc_FFT_values, gyro_FFT_values
    
def preprocess_and_get_output_list(sub_intervals):
    output = []
    for sub_interval in sub_intervals:
        acc_FFT_values, gyro_FFT_values = get_sub_interval_fft(sub_interval)
        append_fft_values_to_output(output, acc_FFT_values)
        append_fft_values_to_output(output, gyro_FFT_values)
    return output
    
def pad_output_list_to_length(output_list, length):
    if len(output_list) < length:
        output_list.extend([0.0 for _ in range(length-len(output_list))])

def append_activity_encoding(output, activity):
    num_activities = 12
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
    dataset_output_folder = "../Data/USC-HAD/Preprocessed"
    sample_index = 0
    create_output_folders(dataset_output_folder, 14, 12)
    
    dataset_folder = "../Data/USC-HAD/USC-HAD"
    for elem in os.listdir(dataset_folder):
        full_elem_path = os.path.join(dataset_folder, elem)
        if os.path.isdir(full_elem_path):
            for mat_file_name in os.listdir(full_elem_path):
                print("Processing file:", mat_file_name)
                og_sensor_readings = get_sensor_reading_from_file(os.path.join(full_elem_path, mat_file_name))
                for sensor_readings in augmented_readings_generator(og_sensor_readings, 10):
                    for sample in get_five_seconds_samples(sensor_readings):
                        sub_intervals = divide_in_sub_intervals(sample, 25)
                        output = preprocess_and_get_output_list(sub_intervals)
                        pad_output_list_to_length(output, 2400)
                        append_activity_encoding(output, get_activity_index(mat_file_name))
                        write_output_to_csv(output, dataset_output_folder, get_subject_index(elem), get_activity_index(mat_file_name), sample_index)
                        sample_index += 1
                        
    print("Num Samples:", sample_index)