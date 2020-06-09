# Attention-Based Deep Learning Framework for Human Activity Recognition with User Adaptation

Reference code for the models presented in the paper:
```
@misc{buffelli2020attentionbased,
    title={Attention-Based Deep Learning Framework for Human Activity Recognition with User Adaptation},
    author={Davide Buffelli and Fabio Vandin},
    year={2020},
    eprint={2006.03820},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
Refer to the paper for a proper presentation of all the models, the preprocessing procedure, details about the datasets, and the training procedure. Please cite the above paper if you use this code in your own work. 

## Instructions
You can train and cross-validate TrASenD, by first inserting the paths to the preprocessed dataset, and the path where the model directories will be saved in the file test\_trasend.py. Then you can run the script with:
```
python test_trasend.py
```
Links to datasets, and requirements can be found below.

In the file usage_example.py you'll find a simple example showing how to train, evaluate, and make predictions with the presented models.

## Datasets
Scripts to preprocess data can be found in the Data Preprocessing folder. 

The original (not preprocessed) datasets can be found at the following links:

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
* Davide Buffelli's Msc Thesis <https://github.com/DavideBuffelli/A-Deep-Learning-Model-for-Personalised-Human-Activity-Recognition>

## License
Refer to the file [LICENSE](LICENSE)
