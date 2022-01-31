# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author: Christoph Hoog Antink, Maurice Rohr
"""

import csv
from re import A, L
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from ecgdetectors import Detectors
import os
from typing import List, Tuple
import os
from scipy.signal.spectral import periodogram

import tensorflow as tf
import tensorflow.keras as keras
from ecgdetectors import Detectors
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models
from keras.models import load_model



###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(ecg_leads : List[np.ndarray], fs : float, ecg_names : List[str], model_name : str='model.npy',is_binary_classifier : bool=False) -> List[Tuple[str,str]]:
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.
    model_name : str
        Name des Models, kann verwendet werden um korrektes Model aus Ordner zu laden
    is_binary_classifier : bool
        Falls getrennte Modelle für F1 und Multi-Score trainiert werden, wird hier übergeben, 
        welches benutzt werden soll
    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''

    #------------------------------------------------------------------------------
    # Euer Code ab hier  
    if (model_name == "CNN_bin"): #Beginn des binären Klassifizierers
        cnn_model = load_model("./CNN_bin/model_bin.hdf5")
        print('Model Loaded!')

        data_names = []
        data_samples = []
        r_peaks_list = []

        detectors = Detectors(fs)  
        for idx, ecg_lead in enumerate(ecg_leads):
            ecg_lead = ecg_lead.astype('float')  # Wandel der Daten von Int in Float32 Format für CNN später
            ecg_lead = (ecg_lead - ecg_lead.mean()) 
            ecg_lead = ecg_lead / (ecg_lead.std() + 1e-08)  
            r_peaks = detectors.hamilton_detector(ecg_lead)     # Detektion der QRS-Komplexe
            sdnn = np.std(np.diff(r_peaks)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden
            for r_peak in r_peaks:
                if r_peak > 150 and r_peak + 450 <= len(ecg_lead):
                    data_samples.append(ecg_lead[r_peak - 150:r_peak + 450])
                    data_names.append(ecg_names[idx])

        data_samples = np.array(data_samples)
        data_samples = data_samples.reshape((*data_samples.shape, 1))

        predictions = list()
        label_predicted = []
        label_predicted_democatric = []
        x = 0
        
        predicted = cnn_model.predict(data_samples) # Hier auch
        print("Test")
        
        for row in predicted:   #Feststellen der wahrscheinlichsten Klasse
            if predicted[x,0]>predicted[x,1]:
                label_predicted.append("N")
            elif predicted[x,0]<predicted[x,1]:
                label_predicted.append("A")
            else:
                print("FEHLER")
            x = x + 1
        n_sum = 0
        a_sum = 0
        t = 0
        for ecg_row in ecg_names:   #Demokratischer Ansatz um EKG-Signale anhand der Herzschlag-Predictions einzuordnen
            for idx, y in enumerate(data_names):
                if (ecg_row==y):
                    if (label_predicted[idx]=='N'):
                        n_sum = n_sum + 1
                    elif (label_predicted[idx]=='A'):
                        a_sum = a_sum +1
                else:
                    pass
            if (n_sum>=a_sum):
                label_predicted_democatric.append("N")
            elif (n_sum<a_sum):
                label_predicted_democatric.append("A")
            print("In {}: Number of A-Heartbeats: {}, Number of N-Heartbeats: {}".format(ecg_row,a_sum, n_sum))
            n_sum = 0
            a_sum = 0       
                    

        
        for idx, name_row in enumerate(ecg_names): #Erstellen des finalen Returnwertes
            predictions.append((ecg_names[idx], label_predicted_democatric[idx]))
        print("fertig")
    elif (model_name == "CNN_multi"): #Beginn des binären Klassifizierers            
            #Beginn des Multilabel-Klassifizierers
            cnn_model = load_model("./CNN_multi/model_multi.hdf5")
            print('Model Loaded!')
            data_names = []
            data_samples = []
            r_peaks_list = []

            detectors = Detectors(fs)  
            for idx, ecg_lead in enumerate(ecg_leads):
                ecg_lead = ecg_lead.astype('float')  # Wandel der Daten von Int in Float32 Format für CNN später
                r_peaks = detectors.hamilton_detector(ecg_lead)     # Detektion der QRS-Komplexe
                sdnn = np.std(np.diff(r_peaks)/fs*1000)             # Berechnung der Standardabweichung der Schlag-zu-Schlag Intervalle (SDNN) in Millisekunden
                for r_peak in r_peaks:
                    if r_peak > 150 and r_peak + 450 <= len(ecg_lead):
                        data_samples.append(ecg_lead[r_peak - 150:r_peak + 450])
                        data_names.append(ecg_names[idx])

            data_samples = np.array(data_samples)
            data_samples = data_samples.reshape((*data_samples.shape, 1))

            predictions = list()
            label_predicted = []
            label_predicted_democatric = []
            x = 0

            predicted = cnn_model.predict(data_samples) # Hier auch

            for row in predicted:   #Feststellen der wahrscheinlichsten Klasse
                if (((predicted[x,0]>predicted[x,1]) and (predicted[x,0]> predicted[x,2]) and (predicted[x,0]>predicted[x,3]))):
                    label_predicted.append("N")
                elif (((predicted[x,1]>predicted[x,0]) and (predicted[x,1]> predicted[x,2]) and (predicted[x,1]>predicted[x,3]))):
                    label_predicted.append("A")
                elif (((predicted[x,2]>predicted[x,0]) and (predicted[x,2]> predicted[x,1]) and (predicted[x,2]>predicted[x,3]))):
                    label_predicted.append("O")
                elif (((predicted[x,3]>predicted[x,0]) and (predicted[x,3]> predicted[x,1]) and (predicted[x,3]>predicted[x,2]))):
                    label_predicted.append("~")
                else:
                    print("FEHLER")
                x = x + 1
            n_sum = 0
            a_sum = 0
            o_sum = 0
            t_sum = 0

            t = 0
            for ecg_row in ecg_names:   #Demokratischer Ansatz um EKG-Signale anhand der Herzschlag-Predictions einzuordnen
                for idx, y in enumerate(data_names):
                    if (ecg_row==y):
                        if (label_predicted[idx]=='N'):
                            n_sum = n_sum + 1
                        elif (label_predicted[idx]=='A'):
                            a_sum = a_sum +1
                        elif (label_predicted[idx]=='O'):
                            o_sum = o_sum +1    
                        elif (label_predicted[idx]=='~'):
                            o_sum = o_sum +1    
                    else:
                        pass
                if ((n_sum>=a_sum)and(n_sum>=o_sum)and(n_sum>=t_sum)):
                    label_predicted_democatric.append("N")
                elif ((a_sum>=n_sum)and(a_sum>=o_sum)and(a_sum>=t_sum)):
                    label_predicted_democatric.append("A")
                elif ((o_sum>=n_sum)and(o_sum>=a_sum)and(o_sum>=t_sum)):
                    label_predicted_democatric.append("O")
                elif ((t_sum>=n_sum)and(t_sum>=o_sum)and(t_sum>=a_sum)):
                    label_predicted_democatric.append("~")
                print("In {}: Number of A-Heartbeats: {}, Number of N-Heartbeats: {}, Number of O-Heartbeats: {}, Number of ~-Heartbeats: {}".format(ecg_row,a_sum, n_sum,o_sum,t_sum))
                n_sum = 0
                a_sum = 0
                o_sum = 0
                t_sum = 0

                        
            for idx, name_row in enumerate(ecg_names): #Erstellen des finalen Returnwertes
                predictions.append((ecg_names[idx], label_predicted_democatric[idx]))
            print("fertig")
                
    #------------------------------------------------------------------------------    
    return predictions # Liste von Tupels im Format (ecg_name,label) - Muss unverändert bleiben!
                               
