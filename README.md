Manual
1.	SVM experiments can be run directly from within Weka 3.6.10 by opening the arff files and selecting the SVM-SMO classifier with default options. 

For Cfs Subset Evaluator, Linear Forward Selection search method must be selected with “numusedattributes=1500” parameter.

Information Gain Attribute Evaluator must be run with default Ranker search method.

2.	RBM experiments are run with Matlab 2019a student version using Tanaka’s Deep Belief Net. Source code is available at the following site:
http://bit.ly/dnnicpr2014

Library files from the above site must be included inside the directory of following source code files, or must be in the Environment path of Matlab.
Matlab Weka library must also be included in the Environment Path of Matlab. This library is needed to read and write arff files from within Matlab and can be download from the following link:
https://www.mathworks.com/matlabcentral/fileexchange/21204-matlab-weka-interface

Matlab main source code files:

all_corpus.m: Matlab source code for EmoDB and EmoSTAR datasets experiments

all_corpus_5mix1.m: Matlab source code for 5Mix dataset experiments

all_corpus_iemo1.m: Matlab source code for IEMOCAP dataset experiments

all_corpus_cfs_ig.m: Matlab source code for Cfs Subset Evaluator and Information Gain Attribute Selector experiments.

all_crosscorpus0.m: Matlab source code for cross-corpus experiments between EmoDb and EmoSTAR.

all_corpus_iemo_cc.m: Matlab source code for cross-corpus experiments between IEMOCAP and 5Mix datasets.

3.	sVGG (small-VGG) experiments are run with Python 3.6.5, Tensorflow 1.9.0, and Keras 2.2.4 on a GPU. SVM, and RBM experiments are reproducible, sVGG experiments are stochastic.

Source code files are listed below.

cnn_speech1d_arff.py: Source code to run single corpus experiments.

cnn_speech1d_arff_cc.py: Source code to run cross-corpus experiments.

Librosa, Matplotlib, Pandas, Plotly, Seaborn, Random, Sys, Os, Glob, sklearn.metrics, Scipy.io.arff libraries must also be installed.

Installation Guide:

Librosa:
pip install librosa
or 
conda install -c conda-forge librosa

Matplotlib:
pip install matplotlib


Pandas:
pip install pandas
or
conda install pandas

Plotly:
pip install plotly
or
conda install -c plotly


Seaborn:
pip install seaborn
or
conda install seaborn

Random:
Import random as rn

Sys:
Import sys

Os:
Import os

Glob:
Import glob

Sklearn.metric:
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

Scipy.io.arff:
import scipy.io as sio
from scipy.io import arff




 
