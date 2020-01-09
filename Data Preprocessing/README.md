# Data Preprocessing 

## HHAR
You can download the original dataset at the following link: <https://archive.ics.uci.edu/ml/datasets/Heterogeneity+Activity+Recognition>.

For the preprocessing we refer to: <https://github.com/DavideBuffelli/A-Deep-Learning-Model-for-Personalised-Human-Activity-Recognition/tree/master/pre-processing>

For an alternative implementation of the preprocessing procedure you can refer to code available from the
authors of DeepSense at: <https://github.com/yscacaca/HHAR-Data-Process>.

## PAMAP2
You can download the original dataset at the following link: <https://archive.ics.uci.edu/ml/datasets/pamap2+physical+activity+monitoring>.

Then follow this procedure (before launching each script make sure to change the directory paths in the code):
1. Launch the script ```convertPAMAP2.py```
2. Launch the script ```organize_for_cv.py```
3. Launch the script ```create_balanced.py```
4. Launch the script ```keep_good_labels_pamap2_.py```

## USC-HAD
You can download the original dataset at the following link: <http://sipi.usc.edu/had/>.

Then follow this procedure (before launching each script make sure to change the directory paths in the code):
1. Launch the script ```convertUSCHAD.py```
2. Launch the script ```organize_for_cv.py```
3. Launch the script ```create_balanced.py```

## Requirements 
* Numpy
* Scipy