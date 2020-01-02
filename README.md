# Attention-Based Deep Learning Framework for Human Activity Recognition with User Adaptation

Reference code for the models presented in the paper:
```
@inproceedings{buffelli2020har,
  title={Attention-Based Deep Learning Framework for Human Activity Recognition with User Adaptation},
  author={Buffelli, Davide and Vandin, Fabio},
  booktitle={venue},
  year={2020}
}
```
Refer to the paper for a proper presentation of all the models, the preprocessing procedure, details about the datasets, and the training procedure. Please cite the above paper if you use this code in your own work. 

## Instructions
You can train and cross-validate TrASenD, by first inserting the paths to the dataset, and the path where the model directories will be saved in the file test\_trasend.py. Then you can run the script with:
```
python test_trasend.py
```
Links to datasets, and requirements can be found below.

In the file usage_example.py you'll find a simple example showing how to train, evaluate, and make predictions with the presented models.

## Datasets
Already Preprocessed datasets con be found at: <>. For an implementation of the preprocessing procedure you can refer to code available from the
authors of DeepSense at: <https://github.com/yscacaca/HHAR-Data-Process>

The original datasets can be found at the following links:

* HHAR
<https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition>

* PAMAP2
<https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring>

* USC-HAD
<http://sipi.usc.edu/had/>

## Requirements

This code has been developed and tested using TensorFlow 1.10. You can install it on your python environment as follows:
```
pip install tensorflow==1.10
```

## Additional Material

* The code released by the authors of DeepSense: <https://github.com/yscacaca/DeepSense>
* ...... my thesis