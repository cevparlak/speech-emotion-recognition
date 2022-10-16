## uses reduce on plateau 
from __future__ import print_function
## Package
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls
import seaborn as sns
import scipy.io.wavfile
import tensorflow as tf
py.init_notebook_mode(connected=True)

import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.callbacks import  History, ReduceLROnPlateau, CSVLogger
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import scipy.io as sio
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy.io import arff
## Python
import random as rn
import sys
from keras.datasets import mnist
from sklearn import preprocessing
import glob
import os
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

alist = pd.DataFrame(columns=['Train', 'Test', 'Acc', 'Loss','Precision','Recall','F1','Kappa','ROC'])

batch_size1 = 32
epoch1 = 500
test_size1=0.30
lr1=0.01;
factor1=0.90
patience1=20
modelstr1='cnn_'+str(lr1)+'_'+str(factor1)+'_'+str(patience1)
strok=0
#modelstr1='cnn'
i=1
np.random.seed(1234)

#session_conf = tf.compat.v1.ConfigProto( intra_op_parallelism_threads=1, inter_op_parallelism_threads=1 )
#sess = tf.compat.v1.Session( graph=tf.compat.v1.get_default_graph(), config=session_conf )
#tf.compat.v1.keras.backend.set_session(sess)

#-----------------------------Keras reproducibility------------------#
SEED = 1234

tf.set_random_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1, 
    inter_op_parallelism_threads=1
)
sess = tf.Session(
    graph=tf.get_default_graph(), 
    config=session_conf
)
K.set_session(sess)
#-----------------------------------------------------------------#
#list1=glob.glob("E:\\PhD\\articles\\Arabian\\converted labels\\*.arff")

path1 = 'E:\\PhD\\articles\\Arabian\\converted labels\\crosscorpus\\'

#list1=['emostarbase-name.arff.arff'    , 'emobase_A_N_H_S-name.arff.arff', 'emostar_STER-name.arff.arff'        , 'emodb_STER_A_N_H_S-name.arff.arff' , 'emostarbase_STER-name.arff.arff'     , 'emobase_STER_A_N_H_S-name.arff.arff'];
#list2=['emobase_A_N_H_S-name.arff.arff', 'emostarbase-name.arff.arff'    , 'emodb_STER_A_N_H_S-name.arff.arff'  , 'emostar_STER-name.arff.arff'       , 'emobase_STER_A_N_H_S-name.arff.arff' , 'emostarbase_STER-name.arff.arff'];

#list1=['iemo_base_AHNS-name.arff.arff' , '5mix_base-name.arff.arff'      , 'iemo_ster_AHNS-name.arff.arff' , '5mix_ster-name.arff.arff'      , 'iemo_base_ster_AHNS-name.arff.arff' , '5mix_base_ster-name.arff.arff'];
#list2=['5mix_base-name.arff.arff'      , 'iemo_base_AHNS-name.arff.arff' , '5mix_ster-name.arff.arff'      , 'iemo_ster_AHNS-name.arff.arff' , '5mix_base_ster-name.arff.arff'      , 'iemo_base_ster_AHNS-name.arff.arff'];

list1=['emostarbase-name.arff.arff'    , 'emobase_A_N_H_S-name.arff.arff', 'emostar_STER-name.arff.arff'        , 'emodb_STER_A_N_H_S-name.arff.arff' , 'emostarbase_STER-name.arff.arff'     , 'emobase_STER_A_N_H_S-name.arff.arff', 'iemo_base_AHNS-name.arff.arff' , '5mix_base-name.arff.arff'      , 'iemo_ster_AHNS-name.arff.arff' , '5mix_ster-name.arff.arff'      , 'iemo_base_ster_AHNS-name.arff.arff' , '5mix_base_ster-name.arff.arff'];
list2=['emobase_A_N_H_S-name.arff.arff', 'emostarbase-name.arff.arff'    , 'emodb_STER_A_N_H_S-name.arff.arff'  , 'emostar_STER-name.arff.arff'       , 'emobase_STER_A_N_H_S-name.arff.arff' , 'emostarbase_STER-name.arff.arff'    , '5mix_base-name.arff.arff'      , 'iemo_base_AHNS-name.arff.arff' , '5mix_ster-name.arff.arff'      , 'iemo_ster_AHNS-name.arff.arff' , '5mix_base_ster-name.arff.arff'      , 'iemo_base_ster_AHNS-name.arff.arff'];

#list1=['emostar_STER-name.arff.arff'      , 'emostar_STER-name.arff.arff' ];
#list2=['emodb_STER_A_N_H_S-name.arff.arff', 'emodb_STER_A_N_H_S-name.arff.arff' ];

#imports
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
import seaborn as sn

def get_new_fig(fn, figsize=[9,9]):
    """ Init graphics """
    plt.clf()    
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()   #Get Current Axis
    ax1.cla() # clear existing plot
    return fig1, ax1
#

def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = []; text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:,col]
    ccl = len(curr_column)

    #last line  and/or last column
    if(col == (ccl - 1)) or (lin == (ccl - 1)):
        #tots and percents
        if(cell_val != 0):
            if(col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%'%(per_ok), '100%'] [per_ok == 100]

        #text to DEL
        text_del.append(oText)

        #text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
        lis_txt = ['%d'%(cell_val), per_ok_s, '%.2f%%'%(per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy(); dic['color'] = 'g'; lis_kwa.append(dic);
        dic = text_kwargs.copy(); dic['color'] = 'r'; lis_kwa.append(dic);
        lis_pos = [(oText._x, oText._y-0.3), (oText._x, oText._y), (oText._x, oText._y+0.3)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            #print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)
        #print '\n'

        #set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if(col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

    else:
        if(per > 0):
            txt = '%s\n%.2f%%' %(cell_val, per)
        else:
            if(show_null_values == 0):
                txt = ''
            elif(show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        #main diagonal
        if(col == lin):
            #set color of the textin the diagonal to white
            oText.set_color('w')
            # set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color('r')

    return text_add, text_del
#

def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append( df_cm[c].sum() )
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append( item_line[1].sum() )
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col
    #print ('\ndf_cm:\n', df_cm, '\n\b\n')
#

def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="Oranges", fmt='.2f', fz=11,
      lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='y'):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    if(pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    #this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    #thanks for seaborn
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

    #set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, fontsize = 10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 25, fontsize = 10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    #face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    #iter in text elements
    array_df = np.array( df_cm.to_records(index=False).tolist() )
    text_add = []; text_del = [];
    posi = -1 #from left to right, bottom to top.
    for t in ax.collections[0].axes.texts: #ax.texts:
        pos = np.array( t.get_position()) - [0.5,0.5]
        lin = int(pos[1]); col = int(pos[0]);
        posi += 1
        #print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        #set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    #remove the old ones
    for item in text_del:
        item.remove()
    #append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    #titles and legends
    ax.set_title('Confusion matrix')
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.tight_layout()  #set layout slim
#    plt.show()
    plt.savefig(head1+'\\conf_mat_'+tail1+'_cnn.png')


def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap="Oranges",
      fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='lin'):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
    """
    from sklearn.metrics import confusion_matrix
    from pandas import DataFrame

    #data
    if(not columns):
        #labels axis integer:
        ##columns = range(1, len(np.unique(y_test))+1)
        #labels axis string:
        from string import ascii_uppercase
        columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]
        columns=['A','H','N','S']
        columns=labels

    confm = confusion_matrix(y_test, predictions)
    cmap = 'Oranges';
    fz = 11;
    figsize=[9,9];
    show_null_values = 2
    df_cm = DataFrame(confm, index=columns, columns=columns)
    pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values, pred_val_axis=pred_val_axis)
#
#TEST functions
def _test_cm():
    #test function with confusion matrix done
    array = np.array( [[13,  0,  1,  0,  2,  0],
                       [ 0, 50,  2,  0, 10,  0],
                       [ 0, 13, 16,  0,  0,  3],
                       [ 0,  0,  0, 13,  1,  0],
                       [ 0, 40,  0,  1, 15,  0],
                       [ 0,  0,  0,  0,  0, 20]])
    #get pandas dataframe
    df_cm = DataFrame(array, index=range(1,7), columns=range(1,7))
    #colormap: see this and choose your more dear
    cmap = 'PuRd'
    pretty_plot_confusion_matrix(df_cm, cmap=cmap)
#
def _test_data_class(y_test1, y_pred1):
    """ test function with y_test (actual values) and predictions (predic) """
    #data
#    y_test = np.array([1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5])
#    predic = np.array([1,2,4,3,5, 1,2,4,3,5, 1,2,3,4,4, 1,4,3,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,3,3,5, 1,2,3,3,5, 1,2,3,4,4, 1,2,3,4,1, 1,2,3,4,1, 1,2,3,4,1, 1,2,4,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5])
    y_test = y_test1
    predic=y_pred1
    """
      Examples to validate output (confusion matrix plot)
        actual: 5 and prediction 1   >>  3
        actual: 2 and prediction 4   >>  1
        actual: 3 and prediction 4   >>  10
    """
    columns = []
    annot = True;
    cmap = 'Oranges';
    fmt = '.2f'
    lw = 0.5
    cbar = False
    show_null_values = 2
    pred_val_axis = 'y'
    #size::
    fz = 12;
    figsize = [9,9];
    if(len(y_test) > 10):
        fz=9; figsize=[14,14];
    plot_confusion_matrix_from_data(y_test, predic, columns,
      annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)


#for st in glob.glob("D:\\emo\\emo1\\meld7emo\\a\\*.arff"):
for i in range(len(list1)):
# load first dataset as train set    
    name1=path1+list1[i]
    print(name1)
    name2=path1+list2[i]
    print(name2)
    df=arff.loadarff(name1)
    df = pd.DataFrame(df[0])
#     here labels are in byte type, we need to convert bytes to integer 
    data=df.drop('class',axis=1)
    df1=df.iloc[:,-1]
#    break
    del df
#    df1[0]=11;
#    df1[1]=22;
    size1=df1.shape[0]
    labels1=np.zeros((size1,1)).astype(int)
    for i1 in range(size1):
        i2=df1[i1] # use this if the labels are 1,2,3,4 in arff file
#        print(i2)
#        df2=int.from_bytes(i2,"little")
#        print(df2)
#        df1[i1]=df2-48
        i3=sys.getsizeof(i2)-34  
        if i3==0:
            labels1[i1]=i2[0]-48
        if i3==1:   # we may have more than 10 classes, upto 99 classes at most
            labels1[i1]=(i2[0]-48)*10+(i2[1]-48)    
        if i3==2:   # we may have more than 100 classes, upto 999 classes at most
            labels1[i1]=(i2[0]-48)*100+(i2[1]-48)*10+(i2[2]-48)    
#        print(i2[0])    
#        print((i2[1]-48)*10+(i2[2]-48)    )
#    del df1
    x=data.values
    x = np.c_[data,labels1]
#    break
#    data = pd.concat([dd1, df1], axis=1)
    y=x[:,-1]
    x=x[:,:-1]
#    break    
    x[np.isnan(x)] = 0
    #    x[~np.all(x == 0, axis=1)]
#    xx=np.any(np.isnan(x))
# normalize each column independently between [0,1]     
    min_max_scaler = preprocessing.MinMaxScaler()
    x= min_max_scaler.fit_transform(x)    
    del data
    # feature count must be greater than 40 or so for the cnn
    nrows=10
    a1=x.shape[1]
    a2=a1 % nrows
    a2=nrows-a2
    c1=np.zeros((x.shape[0],a2),dtype=int)
    x=np.concatenate((x,c1),axis=1)
    values, counts = np.unique(y, return_counts=True)
    n_classes=len(counts)
    ncols=int(x.shape[1]/nrows)
    x_train=x;
    y_train=y;
# load second dataset as test set
    df=arff.loadarff(name2)
    df = pd.DataFrame(df[0])
#     here labels are in byte type, we need to convert bytes to integer 
    data=df.drop('class',axis=1)
    df1=df.iloc[:,-1]
#    break
    del df
#    df1[0]=11;
#    df1[1]=22;
    size1=df1.shape[0]
    labels1=np.zeros((size1,1)).astype(int)
    for i1 in range(size1):
        i2=df1[i1] # use this if the labels are 1,2,3,4 in arff file
#        print(i2)
#        df2=int.from_bytes(i2,"little")
#        print(df2)
#        df1[i1]=df2-48
        i3=sys.getsizeof(i2)-34  
        if i3==0:
            labels1[i1]=i2[0]-48
        if i3==1:   # we may have more than 10 classes, upto 99 classes at most
            labels1[i1]=(i2[0]-48)*10+(i2[1]-48)    
        if i3==2:   # we may have more than 100 classes, upto 999 classes at most
            labels1[i1]=(i2[0]-48)*100+(i2[1]-48)*10+(i2[2]-48)    
#        print(i2[0])    
#        print((i2[1]-48)*10+(i2[2]-48)    )
#    del df1
    x=data.values
    x = np.c_[data,labels1]
#    break
#    data = pd.concat([dd1, df1], axis=1)
    y=x[:,-1]
    x=x[:,:-1]
#    break    
    x[np.isnan(x)] = 0
    #    x[~np.all(x == 0, axis=1)]
#    xx=np.any(np.isnan(x))
# normalize each column independently between [0,1]     
    min_max_scaler = preprocessing.MinMaxScaler()
    x= min_max_scaler.fit_transform(x)    
    del data
    # feature count must be greater than 40 or so for the cnn
    nrows=10
    a1=x.shape[1]
    a2=a1 % nrows
    a2=nrows-a2
    c1=np.zeros((x.shape[0],a2),dtype=int)
    x=np.concatenate((x,c1),axis=1)
    values, counts = np.unique(y, return_counts=True)
    n_classes=len(counts)
    ncols=int(x.shape[1]/nrows)
    x_test=x;
    y_test=y;

    del x
    del y
    
#    return    
    

    # input image dimensions
    img_rows, img_cols = nrows, ncols
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)    
    
    # convert class vectors to binary class matrices
    y_train = y_train-1
    y_test = y_test-1
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')


##3##########################################################
    model = Sequential()								
## or use MiniVGGNet    
    strx1=3
    chanDim = -1
    # first CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(32, (3, 3), padding = "same", input_shape=input_shape))
    model.add(Activation("elu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # second CONV => Relu => CONV => Relu => POOL layer set
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # third CONV => Relu => CONV => Relu => POOL layer set
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # first (and only) set of FC => Relu layers
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # softmax classifier
    model.add(Dense(n_classes, activation='softmax'))

#    optimizer1 = keras.optimizers.SGD(lr=lr1, momentum=0.0, decay=0.0, nesterov=False)
#  # model.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy', fscore])
#    model.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
#  # Model Training
#    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=factor1, patience=patience1, min_lr=0.000000001, verbose=1)
#  # Please change the model name accordingly.
##    mcp_save = ModelCheckpoint('model/aug_noiseNshift_2class2_np.h5', save_best_only=True, monitor='val_loss', mode='min')
#    history=model.fit(x_train, y_train, batch_size = batch_size1, epochs=epoch1,  validation_data=(x_test, y_test), callbacks=[lr_reduce])
#    score = model.evaluate(x_test, y_test, verbose=0)

#      optimizer = Adam(lr=1e-4)
#    optimizer=keras.optimizers.RMSprop(lr=0.0001) #, rho=0.9, epsilon=None, decay=0.0)
#    optimizer=keras.optimizers.Adadelta() #, epsilon=None, decay=0.0)
#  optimizer=keras.optimizers.Adagrad(lr=0.001) #, epsilon=None, decay=0.0)
    optimizer1=keras.optimizers.Adam(lr=lr1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#  optimizer=keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#  optimizer=keras.optimizers.Nadam(lr=0.0001) #, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
    history=model.fit(x_train, y_train, batch_size = batch_size1, epochs=epoch1,  validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)  

    if strok==0:   
        modelstr1=modelstr1+'_Adam_0.01_'+str(strx1)
        strok=1
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    head1, tail1 = os.path.split(name1)
    head2, tail2 = os.path.split(name2)
    tail1=tail1+'---'+tail2
    
# Plotting the Train Valid Loss Graph
    plt.clf()    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'test'])
#    plt.legend(['training', 'test'], loc='upper left')
#    plt.show()
    plt.savefig(head1+'\\aloss_'+tail1+'_'+modelstr1+'.png')

    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train','test'])
#    plt.show()
    plt.savefig(head1+'\\acc_'+tail1+'_'+modelstr1+'.png')

#    plt.clf()
    y_pred = model.predict(x_test, batch_size = batch_size1)
    y_test1=np.argmax(y_test, axis=1)
#    y_test1=y_test1.tolist()
    y_pred1 = np.argmax(y_pred, axis=1)
#    y_pred1 = y_pred1.tolist()
    # predict probabilities for test set
    yhat_probs = y_pred
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(x_test, verbose=0)
    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
#    yhat_classes = yhat_classes[:, 0]
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test1, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test1, yhat_classes,average='weighted')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test1, yhat_classes,average='weighted')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test1, yhat_classes,average='weighted')
    print('F1 score: %f' % f1)
    # kappa
    kappa = cohen_kappa_score(y_test1, yhat_classes)
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
#    auc = roc_auc_score(y_test1, yhat_probs)
#    print('ROC AUC: %f' % auc)
    # confusion matrix
    conf_mat = confusion_matrix(y_test1, yhat_classes)
    df=pd.DataFrame(data=conf_mat[0:,0:], index=[i for i in range(conf_mat.shape[0])], columns=['f'+str(i) for i in range(conf_mat.shape[1])])
    df.to_excel(head1+'\\aconf_mat_'+tail1+'_'+modelstr1+'.xlsx')    
    print(conf_mat)
#    np.savetxt(head1+'\\conf_mat_'+tail1+'_'+modelstr1+'.txt', conf_mat, '%s', delimiter=",")    

#    labels=['ANGRY','HAPPY','NEUTRAL','SAD']
#    print('_test_data_class: test function with y_test (actual values) and predictions (predict)')
#    _test_data_class(y_test1,y_pred1)   
    alist.loc[i]=[tail1,'',score[1],score[0],precision,recall,f1,kappa,'']
    i=i+1
#    np.savetxt(head1+'\\alist_'+'_'+modelstr1+'.csv', alist, '%s', delimiter="/t")
    alist.to_excel(head1+'\\alist_'+'_'+modelstr1+'.xlsx')
    
    
    
    
    
    
    
    
## uses reduce on plateau 

## Package
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls
import seaborn as sns
import scipy.io.wavfile
import tensorflow as tf
py.init_notebook_mode(connected=True)

import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.callbacks import  History, ReduceLROnPlateau, CSVLogger
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import scipy.io as sio
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy.io import arff
## Python
import random as rn
import sys
from keras.datasets import mnist
from sklearn import preprocessing
import glob
import os
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

alist = pd.DataFrame(columns=['Train', 'Test', 'Acc', 'Loss','Precision','Recall','F1','Kappa','ROC'])

batch_size1 = 32
epoch1 = 500
test_size1=0.30
lr1=0.01;
factor1=0.90
patience1=20
modelstr1='cnn_'+str(lr1)+'_'+str(factor1)+'_'+str(patience1)
strok=0
#modelstr1='cnn'
i=1
np.random.seed(1234)

#session_conf = tf.compat.v1.ConfigProto( intra_op_parallelism_threads=1, inter_op_parallelism_threads=1 )
#sess = tf.compat.v1.Session( graph=tf.compat.v1.get_default_graph(), config=session_conf )
#tf.compat.v1.keras.backend.set_session(sess)

#-----------------------------Keras reproducibility------------------#
SEED = 1234

tf.set_random_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1, 
    inter_op_parallelism_threads=1
)
sess = tf.Session(
    graph=tf.get_default_graph(), 
    config=session_conf
)
K.set_session(sess)
#-----------------------------------------------------------------#
#list1=glob.glob("E:\\PhD\\articles\\Arabian\\converted labels\\*.arff")

path1 = 'E:\\PhD\\articles\\Arabian\\converted labels\\crosscorpus\\'

#list1=['emostarbase-name.arff.arff'    , 'emobase_A_N_H_S-name.arff.arff', 'emostar_STER-name.arff.arff'        , 'emodb_STER_A_N_H_S-name.arff.arff' , 'emostarbase_STER-name.arff.arff'     , 'emobase_STER_A_N_H_S-name.arff.arff'];
#list2=['emobase_A_N_H_S-name.arff.arff', 'emostarbase-name.arff.arff'    , 'emodb_STER_A_N_H_S-name.arff.arff'  , 'emostar_STER-name.arff.arff'       , 'emobase_STER_A_N_H_S-name.arff.arff' , 'emostarbase_STER-name.arff.arff'];

#list1=['iemo_base_AHNS-name.arff.arff' , '5mix_base-name.arff.arff'      , 'iemo_ster_AHNS-name.arff.arff' , '5mix_ster-name.arff.arff'      , 'iemo_base_ster_AHNS-name.arff.arff' , '5mix_base_ster-name.arff.arff'];
#list2=['5mix_base-name.arff.arff'      , 'iemo_base_AHNS-name.arff.arff' , '5mix_ster-name.arff.arff'      , 'iemo_ster_AHNS-name.arff.arff' , '5mix_base_ster-name.arff.arff'      , 'iemo_base_ster_AHNS-name.arff.arff'];

list1=['emostarbase-name.arff.arff'    , 'emobase_A_N_H_S-name.arff.arff', 'emostar_STER-name.arff.arff'        , 'emodb_STER_A_N_H_S-name.arff.arff' , 'emostarbase_STER-name.arff.arff'     , 'emobase_STER_A_N_H_S-name.arff.arff', 'iemo_base_AHNS-name.arff.arff' , '5mix_base-name.arff.arff'      , 'iemo_ster_AHNS-name.arff.arff' , '5mix_ster-name.arff.arff'      , 'iemo_base_ster_AHNS-name.arff.arff' , '5mix_base_ster-name.arff.arff'];
list2=['emobase_A_N_H_S-name.arff.arff', 'emostarbase-name.arff.arff'    , 'emodb_STER_A_N_H_S-name.arff.arff'  , 'emostar_STER-name.arff.arff'       , 'emobase_STER_A_N_H_S-name.arff.arff' , 'emostarbase_STER-name.arff.arff'    , '5mix_base-name.arff.arff'      , 'iemo_base_AHNS-name.arff.arff' , '5mix_ster-name.arff.arff'      , 'iemo_ster_AHNS-name.arff.arff' , '5mix_base_ster-name.arff.arff'      , 'iemo_base_ster_AHNS-name.arff.arff'];

#list1=['emostar_STER-name.arff.arff'      , 'emostar_STER-name.arff.arff' ];
#list2=['emodb_STER_A_N_H_S-name.arff.arff', 'emodb_STER_A_N_H_S-name.arff.arff' ];

#imports
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
import seaborn as sn

def get_new_fig(fn, figsize=[9,9]):
    """ Init graphics """
    plt.clf()    
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()   #Get Current Axis
    ax1.cla() # clear existing plot
    return fig1, ax1
#

def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = []; text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:,col]
    ccl = len(curr_column)

    #last line  and/or last column
    if(col == (ccl - 1)) or (lin == (ccl - 1)):
        #tots and percents
        if(cell_val != 0):
            if(col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%'%(per_ok), '100%'] [per_ok == 100]

        #text to DEL
        text_del.append(oText)

        #text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
        lis_txt = ['%d'%(cell_val), per_ok_s, '%.2f%%'%(per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy(); dic['color'] = 'g'; lis_kwa.append(dic);
        dic = text_kwargs.copy(); dic['color'] = 'r'; lis_kwa.append(dic);
        lis_pos = [(oText._x, oText._y-0.3), (oText._x, oText._y), (oText._x, oText._y+0.3)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            #print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)
        #print '\n'

        #set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if(col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

    else:
        if(per > 0):
            txt = '%s\n%.2f%%' %(cell_val, per)
        else:
            if(show_null_values == 0):
                txt = ''
            elif(show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        #main diagonal
        if(col == lin):
            #set color of the textin the diagonal to white
            oText.set_color('w')
            # set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color('r')

    return text_add, text_del
#

def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append( df_cm[c].sum() )
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append( item_line[1].sum() )
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col
    #print ('\ndf_cm:\n', df_cm, '\n\b\n')
#

def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="Oranges", fmt='.2f', fz=11,
      lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='y'):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    if(pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    #this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    #thanks for seaborn
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

    #set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, fontsize = 10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 25, fontsize = 10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    #face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    #iter in text elements
    array_df = np.array( df_cm.to_records(index=False).tolist() )
    text_add = []; text_del = [];
    posi = -1 #from left to right, bottom to top.
    for t in ax.collections[0].axes.texts: #ax.texts:
        pos = np.array( t.get_position()) - [0.5,0.5]
        lin = int(pos[1]); col = int(pos[0]);
        posi += 1
        #print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        #set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    #remove the old ones
    for item in text_del:
        item.remove()
    #append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    #titles and legends
    ax.set_title('Confusion matrix')
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.tight_layout()  #set layout slim
#    plt.show()
    plt.savefig(head1+'\\conf_mat_'+tail1+'_cnn.png')


def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap="Oranges",
      fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='lin'):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
    """
    from sklearn.metrics import confusion_matrix
    from pandas import DataFrame

    #data
    if(not columns):
        #labels axis integer:
        ##columns = range(1, len(np.unique(y_test))+1)
        #labels axis string:
        from string import ascii_uppercase
        columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]
        columns=['A','H','N','S']
        columns=labels

    confm = confusion_matrix(y_test, predictions)
    cmap = 'Oranges';
    fz = 11;
    figsize=[9,9];
    show_null_values = 2
    df_cm = DataFrame(confm, index=columns, columns=columns)
    pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values, pred_val_axis=pred_val_axis)
#
#TEST functions
def _test_cm():
    #test function with confusion matrix done
    array = np.array( [[13,  0,  1,  0,  2,  0],
                       [ 0, 50,  2,  0, 10,  0],
                       [ 0, 13, 16,  0,  0,  3],
                       [ 0,  0,  0, 13,  1,  0],
                       [ 0, 40,  0,  1, 15,  0],
                       [ 0,  0,  0,  0,  0, 20]])
    #get pandas dataframe
    df_cm = DataFrame(array, index=range(1,7), columns=range(1,7))
    #colormap: see this and choose your more dear
    cmap = 'PuRd'
    pretty_plot_confusion_matrix(df_cm, cmap=cmap)
#
def _test_data_class(y_test1, y_pred1):
    """ test function with y_test (actual values) and predictions (predic) """
    #data
#    y_test = np.array([1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5])
#    predic = np.array([1,2,4,3,5, 1,2,4,3,5, 1,2,3,4,4, 1,4,3,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,3,3,5, 1,2,3,3,5, 1,2,3,4,4, 1,2,3,4,1, 1,2,3,4,1, 1,2,3,4,1, 1,2,4,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5])
    y_test = y_test1
    predic=y_pred1
    """
      Examples to validate output (confusion matrix plot)
        actual: 5 and prediction 1   >>  3
        actual: 2 and prediction 4   >>  1
        actual: 3 and prediction 4   >>  10
    """
    columns = []
    annot = True;
    cmap = 'Oranges';
    fmt = '.2f'
    lw = 0.5
    cbar = False
    show_null_values = 2
    pred_val_axis = 'y'
    #size::
    fz = 12;
    figsize = [9,9];
    if(len(y_test) > 10):
        fz=9; figsize=[14,14];
    plot_confusion_matrix_from_data(y_test, predic, columns,
      annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)


#for st in glob.glob("D:\\emo\\emo1\\meld7emo\\a\\*.arff"):
for i in range(len(list1)):
# load first dataset as train set    
    name1=path1+list1[i]
    print(name1)
    name2=path1+list2[i]
    print(name2)
    df=arff.loadarff(name1)
    df = pd.DataFrame(df[0])
#     here labels are in byte type, we need to convert bytes to integer 
    data=df.drop('class',axis=1)
    df1=df.iloc[:,-1]
#    break
    del df
#    df1[0]=11;
#    df1[1]=22;
    size1=df1.shape[0]
    labels1=np.zeros((size1,1)).astype(int)
    for i1 in range(size1):
        i2=df1[i1] # use this if the labels are 1,2,3,4 in arff file
#        print(i2)
#        df2=int.from_bytes(i2,"little")
#        print(df2)
#        df1[i1]=df2-48
        i3=sys.getsizeof(i2)-34  
        if i3==0:
            labels1[i1]=i2[0]-48
        if i3==1:   # we may have more than 10 classes, upto 99 classes at most
            labels1[i1]=(i2[0]-48)*10+(i2[1]-48)    
        if i3==2:   # we may have more than 100 classes, upto 999 classes at most
            labels1[i1]=(i2[0]-48)*100+(i2[1]-48)*10+(i2[2]-48)    
#        print(i2[0])    
#        print((i2[1]-48)*10+(i2[2]-48)    )
#    del df1
    x=data.values
    x = np.c_[data,labels1]
#    break
#    data = pd.concat([dd1, df1], axis=1)
    y=x[:,-1]
    x=x[:,:-1]
#    break    
    x[np.isnan(x)] = 0
    #    x[~np.all(x == 0, axis=1)]
#    xx=np.any(np.isnan(x))
# normalize each column independently between [0,1]     
    min_max_scaler = preprocessing.MinMaxScaler()
    x= min_max_scaler.fit_transform(x)    
    del data
    # feature count must be greater than 40 or so for the cnn
    nrows=10
    a1=x.shape[1]
    a2=a1 % nrows
    a2=nrows-a2
    c1=np.zeros((x.shape[0],a2),dtype=int)
    x=np.concatenate((x,c1),axis=1)
    values, counts = np.unique(y, return_counts=True)
    n_classes=len(counts)
    ncols=int(x.shape[1]/nrows)
    x_train=x;
    y_train=y;
# load second dataset as test set
    df=arff.loadarff(name2)
    df = pd.DataFrame(df[0])
#     here labels are in byte type, we need to convert bytes to integer 
    data=df.drop('class',axis=1)
    df1=df.iloc[:,-1]
#    break
    del df
#    df1[0]=11;
#    df1[1]=22;
    size1=df1.shape[0]
    labels1=np.zeros((size1,1)).astype(int)
    for i1 in range(size1):
        i2=df1[i1] # use this if the labels are 1,2,3,4 in arff file
#        print(i2)
#        df2=int.from_bytes(i2,"little")
#        print(df2)
#        df1[i1]=df2-48
        i3=sys.getsizeof(i2)-34  
        if i3==0:
            labels1[i1]=i2[0]-48
        if i3==1:   # we may have more than 10 classes, upto 99 classes at most
            labels1[i1]=(i2[0]-48)*10+(i2[1]-48)    
        if i3==2:   # we may have more than 100 classes, upto 999 classes at most
            labels1[i1]=(i2[0]-48)*100+(i2[1]-48)*10+(i2[2]-48)    
#        print(i2[0])    
#        print((i2[1]-48)*10+(i2[2]-48)    )
#    del df1
    x=data.values
    x = np.c_[data,labels1]
#    break
#    data = pd.concat([dd1, df1], axis=1)
    y=x[:,-1]
    x=x[:,:-1]
#    break    
    x[np.isnan(x)] = 0
    #    x[~np.all(x == 0, axis=1)]
#    xx=np.any(np.isnan(x))
# normalize each column independently between [0,1]     
    min_max_scaler = preprocessing.MinMaxScaler()
    x= min_max_scaler.fit_transform(x)    
    del data
    # feature count must be greater than 40 or so for the cnn
    nrows=10
    a1=x.shape[1]
    a2=a1 % nrows
    a2=nrows-a2
    c1=np.zeros((x.shape[0],a2),dtype=int)
    x=np.concatenate((x,c1),axis=1)
    values, counts = np.unique(y, return_counts=True)
    n_classes=len(counts)
    ncols=int(x.shape[1]/nrows)
    x_test=x;
    y_test=y;

    del x
    del y
    
#    return    
    

    # input image dimensions
    img_rows, img_cols = nrows, ncols
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)    
    
    # convert class vectors to binary class matrices
    y_train = y_train-1
    y_test = y_test-1
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')


##3##########################################################
    model = Sequential()								
## or use MiniVGGNet    
    strx1=3
    chanDim = -1
    # first CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(32, (3, 3), padding = "same", input_shape=input_shape))
    model.add(Activation("elu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # second CONV => Relu => CONV => Relu => POOL layer set
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # third CONV => Relu => CONV => Relu => POOL layer set
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # first (and only) set of FC => Relu layers
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # softmax classifier
    model.add(Dense(n_classes, activation='softmax'))

#    optimizer1 = keras.optimizers.SGD(lr=lr1, momentum=0.0, decay=0.0, nesterov=False)
#  # model.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy', fscore])
#    model.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
#  # Model Training
#    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=factor1, patience=patience1, min_lr=0.000000001, verbose=1)
#  # Please change the model name accordingly.
##    mcp_save = ModelCheckpoint('model/aug_noiseNshift_2class2_np.h5', save_best_only=True, monitor='val_loss', mode='min')
#    history=model.fit(x_train, y_train, batch_size = batch_size1, epochs=epoch1,  validation_data=(x_test, y_test), callbacks=[lr_reduce])
#    score = model.evaluate(x_test, y_test, verbose=0)

#    optimizer = Adam(lr=1e-4)
#    optimizer=keras.optimizers.RMSprop(lr=0.0001) #, rho=0.9, epsilon=None, decay=0.0)
#    optimizer=keras.optimizers.Adadelta() #, epsilon=None, decay=0.0)
    optimizer1=keras.optimizers.Adagrad(lr=lr1) #, epsilon=None, decay=0.0)
#  # Model Training
#  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#  optimizer=keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#  optimizer=keras.optimizers.Nadam(lr=0.0001) #, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=factor1, patience=patience1, min_lr=0.000000001, verbose=1)
    history=model.fit(x_train, y_train, batch_size = batch_size1, epochs=epoch1,  validation_data=(x_test, y_test), callbacks=[lr_reduce])
    score = model.evaluate(x_test, y_test, verbose=0)  

    if strok==0:   
        modelstr1=modelstr1+'_Adagrad_0.01_'+str(strx1)
        strok=1
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    head1, tail1 = os.path.split(name1)
    head2, tail2 = os.path.split(name2)
    tail1=tail1+'---'+tail2
    
# Plotting the Train Valid Loss Graph
    plt.clf()    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'test'])
#    plt.legend(['training', 'test'], loc='upper left')
#    plt.show()
    plt.savefig(head1+'\\aloss_'+tail1+'_'+modelstr1+'.png')

    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train','test'])
#    plt.show()
    plt.savefig(head1+'\\acc_'+tail1+'_'+modelstr1+'.png')

#    plt.clf()
    y_pred = model.predict(x_test, batch_size = batch_size1)
    y_test1=np.argmax(y_test, axis=1)
#    y_test1=y_test1.tolist()
    y_pred1 = np.argmax(y_pred, axis=1)
#    y_pred1 = y_pred1.tolist()
    # predict probabilities for test set
    yhat_probs = y_pred
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(x_test, verbose=0)
    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
#    yhat_classes = yhat_classes[:, 0]
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test1, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test1, yhat_classes,average='weighted')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test1, yhat_classes,average='weighted')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test1, yhat_classes,average='weighted')
    print('F1 score: %f' % f1)
    # kappa
    kappa = cohen_kappa_score(y_test1, yhat_classes)
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
#    auc = roc_auc_score(y_test1, yhat_probs)
#    print('ROC AUC: %f' % auc)
    # confusion matrix
    conf_mat = confusion_matrix(y_test1, yhat_classes)
    df=pd.DataFrame(data=conf_mat[0:,0:], index=[i for i in range(conf_mat.shape[0])], columns=['f'+str(i) for i in range(conf_mat.shape[1])])
    df.to_excel(head1+'\\aconf_mat_'+tail1+'_'+modelstr1+'.xlsx')    
    print(conf_mat)
#    np.savetxt(head1+'\\conf_mat_'+tail1+'_'+modelstr1+'.txt', conf_mat, '%s', delimiter=",")    

#    labels=['ANGRY','HAPPY','NEUTRAL','SAD']
#    print('_test_data_class: test function with y_test (actual values) and predictions (predict)')
#    _test_data_class(y_test1,y_pred1)   
    alist.loc[i]=[tail1,'',score[1],score[0],precision,recall,f1,kappa,'']
    i=i+1
#    np.savetxt(head1+'\\alist_'+'_'+modelstr1+'.csv', alist, '%s', delimiter="/t")
    alist.to_excel(head1+'\\alist_'+'_'+modelstr1+'.xlsx')
        
    
    
    
    
    
    
    
    
    
    
    
## uses reduce on plateau 

## Package
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as py
import plotly.tools as tls
import seaborn as sns
import scipy.io.wavfile
import tensorflow as tf
py.init_notebook_mode(connected=True)

import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras.callbacks import  History, ReduceLROnPlateau, CSVLogger
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import scipy.io as sio
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy.io import arff
## Python
import random as rn
import sys
from keras.datasets import mnist
from sklearn import preprocessing
import glob
import os
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

alist = pd.DataFrame(columns=['Train', 'Test', 'Acc', 'Loss','Precision','Recall','F1','Kappa','ROC'])

batch_size1 = 32
epoch1 = 500
test_size1=0.30
lr1=0.01;
factor1=0.90
patience1=20
modelstr1='cnn_'+str(lr1)+'_'+str(factor1)+'_'+str(patience1)
strok=0
#modelstr1='cnn'
i=1
np.random.seed(1234)

#session_conf = tf.compat.v1.ConfigProto( intra_op_parallelism_threads=1, inter_op_parallelism_threads=1 )
#sess = tf.compat.v1.Session( graph=tf.compat.v1.get_default_graph(), config=session_conf )
#tf.compat.v1.keras.backend.set_session(sess)

#-----------------------------Keras reproducibility------------------#
SEED = 1234

tf.set_random_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1, 
    inter_op_parallelism_threads=1
)
sess = tf.Session(
    graph=tf.get_default_graph(), 
    config=session_conf
)
K.set_session(sess)
#-----------------------------------------------------------------#
#list1=glob.glob("E:\\PhD\\articles\\Arabian\\converted labels\\*.arff")

path1 = 'E:\\PhD\\articles\\Arabian\\converted labels\\crosscorpus\\'

#list1=['emostarbase-name.arff.arff'    , 'emobase_A_N_H_S-name.arff.arff', 'emostar_STER-name.arff.arff'        , 'emodb_STER_A_N_H_S-name.arff.arff' , 'emostarbase_STER-name.arff.arff'     , 'emobase_STER_A_N_H_S-name.arff.arff'];
#list2=['emobase_A_N_H_S-name.arff.arff', 'emostarbase-name.arff.arff'    , 'emodb_STER_A_N_H_S-name.arff.arff'  , 'emostar_STER-name.arff.arff'       , 'emobase_STER_A_N_H_S-name.arff.arff' , 'emostarbase_STER-name.arff.arff'];

#list1=['iemo_base_AHNS-name.arff.arff' , '5mix_base-name.arff.arff'      , 'iemo_ster_AHNS-name.arff.arff' , '5mix_ster-name.arff.arff'      , 'iemo_base_ster_AHNS-name.arff.arff' , '5mix_base_ster-name.arff.arff'];
#list2=['5mix_base-name.arff.arff'      , 'iemo_base_AHNS-name.arff.arff' , '5mix_ster-name.arff.arff'      , 'iemo_ster_AHNS-name.arff.arff' , '5mix_base_ster-name.arff.arff'      , 'iemo_base_ster_AHNS-name.arff.arff'];

list1=['emostarbase-name.arff.arff'    , 'emobase_A_N_H_S-name.arff.arff', 'emostar_STER-name.arff.arff'        , 'emodb_STER_A_N_H_S-name.arff.arff' , 'emostarbase_STER-name.arff.arff'     , 'emobase_STER_A_N_H_S-name.arff.arff', 'iemo_base_AHNS-name.arff.arff' , '5mix_base-name.arff.arff'      , 'iemo_ster_AHNS-name.arff.arff' , '5mix_ster-name.arff.arff'      , 'iemo_base_ster_AHNS-name.arff.arff' , '5mix_base_ster-name.arff.arff'];
list2=['emobase_A_N_H_S-name.arff.arff', 'emostarbase-name.arff.arff'    , 'emodb_STER_A_N_H_S-name.arff.arff'  , 'emostar_STER-name.arff.arff'       , 'emobase_STER_A_N_H_S-name.arff.arff' , 'emostarbase_STER-name.arff.arff'    , '5mix_base-name.arff.arff'      , 'iemo_base_AHNS-name.arff.arff' , '5mix_ster-name.arff.arff'      , 'iemo_ster_AHNS-name.arff.arff' , '5mix_base_ster-name.arff.arff'      , 'iemo_base_ster_AHNS-name.arff.arff'];

#list1=['emostar_STER-name.arff.arff'      , 'emostar_STER-name.arff.arff' ];
#list2=['emodb_STER_A_N_H_S-name.arff.arff', 'emodb_STER_A_N_H_S-name.arff.arff' ];

#imports
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
import seaborn as sn

def get_new_fig(fn, figsize=[9,9]):
    """ Init graphics """
    plt.clf()    
    fig1 = plt.figure(fn, figsize)
    ax1 = fig1.gca()   #Get Current Axis
    ax1.cla() # clear existing plot
    return fig1, ax1
#

def configcell_text_and_colors(array_df, lin, col, oText, facecolors, posi, fz, fmt, show_null_values=0):
    """
      config cell text and colors
      and return text elements to add and to dell
      @TODO: use fmt
    """
    text_add = []; text_del = [];
    cell_val = array_df[lin][col]
    tot_all = array_df[-1][-1]
    per = (float(cell_val) / tot_all) * 100
    curr_column = array_df[:,col]
    ccl = len(curr_column)

    #last line  and/or last column
    if(col == (ccl - 1)) or (lin == (ccl - 1)):
        #tots and percents
        if(cell_val != 0):
            if(col == ccl - 1) and (lin == ccl - 1):
                tot_rig = 0
                for i in range(array_df.shape[0] - 1):
                    tot_rig += array_df[i][i]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(col == ccl - 1):
                tot_rig = array_df[lin][lin]
                per_ok = (float(tot_rig) / cell_val) * 100
            elif(lin == ccl - 1):
                tot_rig = array_df[col][col]
                per_ok = (float(tot_rig) / cell_val) * 100
            per_err = 100 - per_ok
        else:
            per_ok = per_err = 0

        per_ok_s = ['%.2f%%'%(per_ok), '100%'] [per_ok == 100]

        #text to DEL
        text_del.append(oText)

        #text to ADD
        font_prop = fm.FontProperties(weight='bold', size=fz)
        text_kwargs = dict(color='w', ha="center", va="center", gid='sum', fontproperties=font_prop)
        lis_txt = ['%d'%(cell_val), per_ok_s, '%.2f%%'%(per_err)]
        lis_kwa = [text_kwargs]
        dic = text_kwargs.copy(); dic['color'] = 'g'; lis_kwa.append(dic);
        dic = text_kwargs.copy(); dic['color'] = 'r'; lis_kwa.append(dic);
        lis_pos = [(oText._x, oText._y-0.3), (oText._x, oText._y), (oText._x, oText._y+0.3)]
        for i in range(len(lis_txt)):
            newText = dict(x=lis_pos[i][0], y=lis_pos[i][1], text=lis_txt[i], kw=lis_kwa[i])
            #print 'lin: %s, col: %s, newText: %s' %(lin, col, newText)
            text_add.append(newText)
        #print '\n'

        #set background color for sum cells (last line and last column)
        carr = [0.27, 0.30, 0.27, 1.0]
        if(col == ccl - 1) and (lin == ccl - 1):
            carr = [0.17, 0.20, 0.17, 1.0]
        facecolors[posi] = carr

    else:
        if(per > 0):
            txt = '%s\n%.2f%%' %(cell_val, per)
        else:
            if(show_null_values == 0):
                txt = ''
            elif(show_null_values == 1):
                txt = '0'
            else:
                txt = '0\n0.0%'
        oText.set_text(txt)

        #main diagonal
        if(col == lin):
            #set color of the textin the diagonal to white
            oText.set_color('w')
            # set background color in the diagonal to blue
            facecolors[posi] = [0.35, 0.8, 0.55, 1.0]
        else:
            oText.set_color('r')

    return text_add, text_del
#

def insert_totals(df_cm):
    """ insert total column and line (the last ones) """
    sum_col = []
    for c in df_cm.columns:
        sum_col.append( df_cm[c].sum() )
    sum_lin = []
    for item_line in df_cm.iterrows():
        sum_lin.append( item_line[1].sum() )
    df_cm['sum_lin'] = sum_lin
    sum_col.append(np.sum(sum_lin))
    df_cm.loc['sum_col'] = sum_col
    #print ('\ndf_cm:\n', df_cm, '\n\b\n')
#

def pretty_plot_confusion_matrix(df_cm, annot=True, cmap="Oranges", fmt='.2f', fz=11,
      lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='y'):
    """
      print conf matrix with default layout (like matlab)
      params:
        df_cm          dataframe (pandas) without totals
        annot          print text in each cell
        cmap           Oranges,Oranges_r,YlGnBu,Blues,RdBu, ... see:
        fz             fontsize
        lw             linewidth
        pred_val_axis  where to show the prediction values (x or y axis)
                        'col' or 'x': show predicted values in columns (x axis) instead lines
                        'lin' or 'y': show predicted values in lines   (y axis)
    """
    if(pred_val_axis in ('col', 'x')):
        xlbl = 'Predicted'
        ylbl = 'Actual'
    else:
        xlbl = 'Actual'
        ylbl = 'Predicted'
        df_cm = df_cm.T

    # create "Total" column
    insert_totals(df_cm)

    #this is for print allways in the same window
    fig, ax1 = get_new_fig('Conf matrix default', figsize)

    #thanks for seaborn
    ax = sn.heatmap(df_cm, annot=annot, annot_kws={"size": fz}, linewidths=lw, ax=ax1,
                    cbar=cbar, cmap=cmap, linecolor='w', fmt=fmt)

    #set ticklabels rotation
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, fontsize = 10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 25, fontsize = 10)

    # Turn off all the ticks
    for t in ax.xaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False
    for t in ax.yaxis.get_major_ticks():
        t.tick1On = False
        t.tick2On = False

    #face colors list
    quadmesh = ax.findobj(QuadMesh)[0]
    facecolors = quadmesh.get_facecolors()

    #iter in text elements
    array_df = np.array( df_cm.to_records(index=False).tolist() )
    text_add = []; text_del = [];
    posi = -1 #from left to right, bottom to top.
    for t in ax.collections[0].axes.texts: #ax.texts:
        pos = np.array( t.get_position()) - [0.5,0.5]
        lin = int(pos[1]); col = int(pos[0]);
        posi += 1
        #print ('>>> pos: %s, posi: %s, val: %s, txt: %s' %(pos, posi, array_df[lin][col], t.get_text()))

        #set text
        txt_res = configcell_text_and_colors(array_df, lin, col, t, facecolors, posi, fz, fmt, show_null_values)

        text_add.extend(txt_res[0])
        text_del.extend(txt_res[1])

    #remove the old ones
    for item in text_del:
        item.remove()
    #append the new ones
    for item in text_add:
        ax.text(item['x'], item['y'], item['text'], **item['kw'])

    #titles and legends
    ax.set_title('Confusion matrix')
    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    plt.tight_layout()  #set layout slim
#    plt.show()
    plt.savefig(head1+'\\conf_mat_'+tail1+'_cnn.png')


def plot_confusion_matrix_from_data(y_test, predictions, columns=None, annot=True, cmap="Oranges",
      fmt='.2f', fz=11, lw=0.5, cbar=False, figsize=[8,8], show_null_values=0, pred_val_axis='lin'):
    """
        plot confusion matrix function with y_test (actual values) and predictions (predic),
        whitout a confusion matrix yet
    """
    from sklearn.metrics import confusion_matrix
    from pandas import DataFrame

    #data
    if(not columns):
        #labels axis integer:
        ##columns = range(1, len(np.unique(y_test))+1)
        #labels axis string:
        from string import ascii_uppercase
        columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(y_test))]]
        columns=['A','H','N','S']
        columns=labels

    confm = confusion_matrix(y_test, predictions)
    cmap = 'Oranges';
    fz = 11;
    figsize=[9,9];
    show_null_values = 2
    df_cm = DataFrame(confm, index=columns, columns=columns)
    pretty_plot_confusion_matrix(df_cm, fz=fz, cmap=cmap, figsize=figsize, show_null_values=show_null_values, pred_val_axis=pred_val_axis)
#
#TEST functions
def _test_cm():
    #test function with confusion matrix done
    array = np.array( [[13,  0,  1,  0,  2,  0],
                       [ 0, 50,  2,  0, 10,  0],
                       [ 0, 13, 16,  0,  0,  3],
                       [ 0,  0,  0, 13,  1,  0],
                       [ 0, 40,  0,  1, 15,  0],
                       [ 0,  0,  0,  0,  0, 20]])
    #get pandas dataframe
    df_cm = DataFrame(array, index=range(1,7), columns=range(1,7))
    #colormap: see this and choose your more dear
    cmap = 'PuRd'
    pretty_plot_confusion_matrix(df_cm, cmap=cmap)
#
def _test_data_class(y_test1, y_pred1):
    """ test function with y_test (actual values) and predictions (predic) """
    #data
#    y_test = np.array([1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5])
#    predic = np.array([1,2,4,3,5, 1,2,4,3,5, 1,2,3,4,4, 1,4,3,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,3,3,5, 1,2,3,3,5, 1,2,3,4,4, 1,2,3,4,1, 1,2,3,4,1, 1,2,3,4,1, 1,2,4,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,4,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5])
    y_test = y_test1
    predic=y_pred1
    """
      Examples to validate output (confusion matrix plot)
        actual: 5 and prediction 1   >>  3
        actual: 2 and prediction 4   >>  1
        actual: 3 and prediction 4   >>  10
    """
    columns = []
    annot = True;
    cmap = 'Oranges';
    fmt = '.2f'
    lw = 0.5
    cbar = False
    show_null_values = 2
    pred_val_axis = 'y'
    #size::
    fz = 12;
    figsize = [9,9];
    if(len(y_test) > 10):
        fz=9; figsize=[14,14];
    plot_confusion_matrix_from_data(y_test, predic, columns,
      annot, cmap, fmt, fz, lw, cbar, figsize, show_null_values, pred_val_axis)


#for st in glob.glob("D:\\emo\\emo1\\meld7emo\\a\\*.arff"):
for i in range(len(list1)):
# load first dataset as train set    
    name1=path1+list1[i]
    print(name1)
    name2=path1+list2[i]
    print(name2)
    df=arff.loadarff(name1)
    df = pd.DataFrame(df[0])
#     here labels are in byte type, we need to convert bytes to integer 
    data=df.drop('class',axis=1)
    df1=df.iloc[:,-1]
#    break
    del df
#    df1[0]=11;
#    df1[1]=22;
    size1=df1.shape[0]
    labels1=np.zeros((size1,1)).astype(int)
    for i1 in range(size1):
        i2=df1[i1] # use this if the labels are 1,2,3,4 in arff file
#        print(i2)
#        df2=int.from_bytes(i2,"little")
#        print(df2)
#        df1[i1]=df2-48
        i3=sys.getsizeof(i2)-34  
        if i3==0:
            labels1[i1]=i2[0]-48
        if i3==1:   # we may have more than 10 classes, upto 99 classes at most
            labels1[i1]=(i2[0]-48)*10+(i2[1]-48)    
        if i3==2:   # we may have more than 100 classes, upto 999 classes at most
            labels1[i1]=(i2[0]-48)*100+(i2[1]-48)*10+(i2[2]-48)    
#        print(i2[0])    
#        print((i2[1]-48)*10+(i2[2]-48)    )
#    del df1
    x=data.values
    x = np.c_[data,labels1]
#    break
#    data = pd.concat([dd1, df1], axis=1)
    y=x[:,-1]
    x=x[:,:-1]
#    break    
    x[np.isnan(x)] = 0
    #    x[~np.all(x == 0, axis=1)]
#    xx=np.any(np.isnan(x))
# normalize each column independently between [0,1]     
    min_max_scaler = preprocessing.MinMaxScaler()
    x= min_max_scaler.fit_transform(x)    
    del data
    # feature count must be greater than 40 or so for the cnn
    nrows=10
    a1=x.shape[1]
    a2=a1 % nrows
    a2=nrows-a2
    c1=np.zeros((x.shape[0],a2),dtype=int)
    x=np.concatenate((x,c1),axis=1)
    values, counts = np.unique(y, return_counts=True)
    n_classes=len(counts)
    ncols=int(x.shape[1]/nrows)
    x_train=x;
    y_train=y;
# load second dataset as test set
    df=arff.loadarff(name2)
    df = pd.DataFrame(df[0])
#     here labels are in byte type, we need to convert bytes to integer 
    data=df.drop('class',axis=1)
    df1=df.iloc[:,-1]
#    break
    del df
#    df1[0]=11;
#    df1[1]=22;
    size1=df1.shape[0]
    labels1=np.zeros((size1,1)).astype(int)
    for i1 in range(size1):
        i2=df1[i1] # use this if the labels are 1,2,3,4 in arff file
#        print(i2)
#        df2=int.from_bytes(i2,"little")
#        print(df2)
#        df1[i1]=df2-48
        i3=sys.getsizeof(i2)-34  
        if i3==0:
            labels1[i1]=i2[0]-48
        if i3==1:   # we may have more than 10 classes, upto 99 classes at most
            labels1[i1]=(i2[0]-48)*10+(i2[1]-48)    
        if i3==2:   # we may have more than 100 classes, upto 999 classes at most
            labels1[i1]=(i2[0]-48)*100+(i2[1]-48)*10+(i2[2]-48)    
#        print(i2[0])    
#        print((i2[1]-48)*10+(i2[2]-48)    )
#    del df1
    x=data.values
    x = np.c_[data,labels1]
#    break
#    data = pd.concat([dd1, df1], axis=1)
    y=x[:,-1]
    x=x[:,:-1]
#    break    
    x[np.isnan(x)] = 0
    #    x[~np.all(x == 0, axis=1)]
#    xx=np.any(np.isnan(x))
# normalize each column independently between [0,1]     
    min_max_scaler = preprocessing.MinMaxScaler()
    x= min_max_scaler.fit_transform(x)    
    del data
    # feature count must be greater than 40 or so for the cnn
    nrows=10
    a1=x.shape[1]
    a2=a1 % nrows
    a2=nrows-a2
    c1=np.zeros((x.shape[0],a2),dtype=int)
    x=np.concatenate((x,c1),axis=1)
    values, counts = np.unique(y, return_counts=True)
    n_classes=len(counts)
    ncols=int(x.shape[1]/nrows)
    x_test=x;
    y_test=y;

    del x
    del y
    
#    return    
    

    # input image dimensions
    img_rows, img_cols = nrows, ncols
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)    
    
    # convert class vectors to binary class matrices
    y_train = y_train-1
    y_test = y_test-1
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)
    
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')


##3##########################################################
    model = Sequential()								
## or use MiniVGGNet    
    strx1=3
    chanDim = -1
    # first CONV => RELU => CONV => RELU => POOL layer set
    model.add(Conv2D(32, (3, 3), padding = "same", input_shape=input_shape))
    model.add(Activation("elu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # second CONV => Relu => CONV => Relu => POOL layer set
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # third CONV => Relu => CONV => Relu => POOL layer set
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # first (and only) set of FC => Relu layers
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("elu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # softmax classifier
    model.add(Dense(n_classes, activation='softmax'))

#    optimizer1 = keras.optimizers.SGD(lr=lr1, momentum=0.0, decay=0.0, nesterov=False)
#  # model.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy', fscore])
#    model.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
#  # Model Training
#    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=factor1, patience=patience1, min_lr=0.000000001, verbose=1)
#  # Please change the model name accordingly.
##    mcp_save = ModelCheckpoint('model/aug_noiseNshift_2class2_np.h5', save_best_only=True, monitor='val_loss', mode='min')
#    history=model.fit(x_train, y_train, batch_size = batch_size1, epochs=epoch1,  validation_data=(x_test, y_test), callbacks=[lr_reduce])
#    score = model.evaluate(x_test, y_test, verbose=0)


#    optimizer = Adam(lr=1e-4)
#    optimizer=keras.optimizers.RMSprop(lr=0.0001) #, rho=0.9, epsilon=None, decay=0.0)
#    optimizer=keras.optimizers.Adadelta() #, epsilon=None, decay=0.0)
    optimizer1=keras.optimizers.Adadelta(lr=lr1) #, epsilon=None, decay=0.0)
#  # Model Training
#  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#  optimizer=keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#  optimizer=keras.optimizers.Nadam(lr=0.0001) #, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=factor1, patience=patience1, min_lr=0.000000001, verbose=1)
    history=model.fit(x_train, y_train, batch_size = batch_size1, epochs=epoch1,  validation_data=(x_test, y_test), callbacks=[lr_reduce])
    score = model.evaluate(x_test, y_test, verbose=0)  


    if strok==0:   
        modelstr1=modelstr1+'_Adadelta_0.01_'+str(strx1)
        strok=1
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    head1, tail1 = os.path.split(name1)
    head2, tail2 = os.path.split(name2)
    tail1=tail1+'---'+tail2
    
# Plotting the Train Valid Loss Graph
    plt.clf()    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'test'])
#    plt.legend(['training', 'test'], loc='upper left')
#    plt.show()
    plt.savefig(head1+'\\aloss_'+tail1+'_'+modelstr1+'.png')

    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['train','test'])
#    plt.show()
    plt.savefig(head1+'\\acc_'+tail1+'_'+modelstr1+'.png')

#    plt.clf()
    y_pred = model.predict(x_test, batch_size = batch_size1)
    y_test1=np.argmax(y_test, axis=1)
#    y_test1=y_test1.tolist()
    y_pred1 = np.argmax(y_pred, axis=1)
#    y_pred1 = y_pred1.tolist()
    # predict probabilities for test set
    yhat_probs = y_pred
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(x_test, verbose=0)
    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
#    yhat_classes = yhat_classes[:, 0]
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test1, yhat_classes)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test1, yhat_classes,average='weighted')
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test1, yhat_classes,average='weighted')
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test1, yhat_classes,average='weighted')
    print('F1 score: %f' % f1)
    # kappa
    kappa = cohen_kappa_score(y_test1, yhat_classes)
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
#    auc = roc_auc_score(y_test1, yhat_probs)
#    print('ROC AUC: %f' % auc)
    # confusion matrix
    conf_mat = confusion_matrix(y_test1, yhat_classes)
    df=pd.DataFrame(data=conf_mat[0:,0:], index=[i for i in range(conf_mat.shape[0])], columns=['f'+str(i) for i in range(conf_mat.shape[1])])
    df.to_excel(head1+'\\aconf_mat_'+tail1+'_'+modelstr1+'.xlsx')    
    print(conf_mat)
#    np.savetxt(head1+'\\conf_mat_'+tail1+'_'+modelstr1+'.txt', conf_mat, '%s', delimiter=",")    

#    labels=['ANGRY','HAPPY','NEUTRAL','SAD']
#    print('_test_data_class: test function with y_test (actual values) and predictions (predict)')
#    _test_data_class(y_test1,y_pred1)   
    alist.loc[i]=[tail1,'',score[1],score[0],precision,recall,f1,kappa,'']
    i=i+1
#    np.savetxt(head1+'\\alist_'+'_'+modelstr1+'.csv', alist, '%s', delimiter="/t")
    alist.to_excel(head1+'\\alist_'+'_'+modelstr1+'.xlsx')
            