from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import buttord, butter, lfilter, filtfilt, argrelextrema,argrelmax,argrelmin,find_peaks_cwt
from math import pi
from scipy.signal import freqz
from sklearn.externals import joblib
import csv
import warnings

def butter_bandpass(wave_input):
    globals()
    order = 1
    wave = wave_input
    fs = 170
    lowcut = 0.5 #0.5
    highcut = 6 #6
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=0)
    y = filtfilt(b, a, wave)
    return y

def systolic_feature_points(wave,):
    maxi = argrelmax(wave)
    mini = argrelmin(wave)
    maxidata = []
    mx, mn = [], []
    for i in range(len(wave)):
        if (i in maxi[0]) :
            maxidata.append(wave[i])
            mx.append(wave[i])
        else:
            maxidata.append(None)
    mxpos = []
    mxpos = maxi[0]
    mxpos = mxpos.tolist()
    res = [i for i in maxidata if i]
    return mxpos, maxidata, abs(np.mean(res)), abs(np.median(res)), abs(max(res)), min(res) 

def diastolic_ppg_features(wave_input):
    wave = wave_input
    mean = np.mean(np.asarray(wave))
    diff = np.diff(wave, n = 1)
    maxi = argrelmin(diff) #extrema(diff, np.greater)
    maxidata = []
    fp_dat = []
    indx = []
    fpdat_proc = []
    for i in range(len(diff)):
        if (i in maxi[0]):
            fp_dat.append(wave[i-1])
            if (wave[i-1] < (mean)):
                indx.append(i-1)
                fpdat_proc.append(wave[i-1])
            else:
                fpdat_proc.append(None)
        else:
            fp_dat.append(None)
            fpdat_proc.append(None)
    mxpos = []
    mxpos = maxi[0]
    mxpos = mxpos.tolist()
    res = [i for i in fpdat_proc if i]
    return indx, fpdat_proc, abs(np.mean(res)), abs(np.median(res)), abs(max(res)), min(res) 

feat_list = []
filename=input("Enter the signal filename with extension.txt here:   ")
print("Filtering Signal, please wait...")
warnings.filterwarnings("ignore")
fp = open(filename, 'r')
ppg= []
rawdata = fp.read().splitlines()
for each in rawdata:
    each = each#[1:-1]
    each_1 = each.split(',')
    ppg.append(int(each_1[0]))
ppg = ppg
output = butter_bandpass(ppg)
output2 = butter_bandpass(ppg)
output2 = butter_bandpass(output2)
sys_fp2, sys_feature_pts2, mean_peak, median_peak, max_peak, min_peak = systolic_feature_points(output2) 
dia_fp2, dia_feature_pts2, mean_notch, median_notch, max_notch, min_notch = diastolic_ppg_features(output)

print("Getting features, please wait...")
feats = []
feats.append(mean_peak)
feats.append(median_peak)
feats.append(max_peak)
feats.append(min_peak)
feats.append(mean_notch)
feats.append(median_notch)
feats.append(max_notch)
feats.append(min_notch)
feat_list.append(feats)

option=input("Use only ANN?... Y/N:   ")
if option=='Y':
    print("Running NeuralNet, please wait...")
    filename_list = ['ann_trained.sav']
    ann_phrase = "The Neural Network Predicts that:"
    for filename in filename_list:
        phrase = ann_phrase
        loaded_model = joblib.load(filename)
        result = loaded_model.predict(feat_list)
        print(result)
        print(phrase)
        if(result==1):
            print("Ailment found in subject")
        elif(result==0):
            print("Healthy Subject")
        else:
            print("Mystery illness, you broke our code!")
else:
    print("Running classifiers, please wait...")
    filename_list = ['svm_trained.sav','dt_gini_trained.sav','dt_entropy_trained.sav','ann_trained.sav']
    ann_phrase = "The Neural Network Predicts that:"
    dt_gini_phrase = "The Decision Tree (Gini) Predicts that:"
    dt_entropy_phrase = "The Decision Tree (IG) Predicts that:"
    svm_phrase = "The SVM Predicts that:"
    for filename in filename_list:
        if filename=='ann_trained.sav':
            phrase=ann_phrase
        elif filename=='dt_gini_trained.sav':
            phrase=dt_gini_phrase
        elif filename=='dt_entropy_trained.sav':
            phrase=dt_entropy_phrase
        elif filename=='svm_trained.sav':
            phrase=svm_phrase
        loaded_model = joblib.load(filename)
        result = loaded_model.predict(feat_list)
        print(result)
        print(phrase)
        if(result==1):
            print("Ailment found in subject")
        elif(result==0):
            print("Healthy Subject")
        else:
            print("Mystery illness, you broke our code!")