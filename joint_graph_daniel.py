#import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#### now by structure and dimension: 116*4
#### for pair, l = 494*1, h = 1*1 -> multiple to get result
### 494 patients

def find_max_ratio_byAll(normal):
    max_ratio_byAll = 0    
    arr = np.zeros((116,4))
    
    for i in range(4):
        for j in range(116):
            all_pat_byDimStruct = np.zeros(len(combined_group))
            normal_pat_byDimStruct = np.zeros(len(normal))
            AD_pat_byDimStruct = np.zeros((len(combined_group)-len(normal)))
            
            normal_counter = 0
            AD_counter = 0
            
            h_scalar = h[j,i]
            for p in range(len(combined_group)): ##in range 494, first set all the values by dim+struct per patient
                pat = combined_group[p]
                l_scalar = l[p,i]
                all_pat_byDimStruct[p] = l_scalar*h_scalar
                
            for k in range(len(combined_group)): ##now find SSE for normal, AD and combined
                pati = combined_group[k]
                if pati in normal:
                    normal_pat_byDimStruct[normal_counter] = all_pat_byDimStruct[k]
                    normal_counter += 1
                else:
                    AD_pat_byDimStruct[AD_counter] = all_pat_byDimStruct[k]
                    AD_counter += 1
            
            normal_SSE = np.sum(np.square(np.subtract(normal_pat_byDimStruct, np.mean(normal_pat_byDimStruct))))
            AD_SSE = np.sum(np.square(np.subtract(AD_pat_byDimStruct, np.mean(AD_pat_byDimStruct))))
            combined_SSE = np.sum(np.square(np.subtract(all_pat_byDimStruct, np.mean(all_pat_byDimStruct))))
            
            ratio = combined_SSE/(normal_SSE + AD_SSE)
            arr[j,i] = ratio
            if ratio > max_ratio_byAll:
                max_ratio_byAll = ratio
    
    return max_ratio_byAll, arr

if __name__=="__main__":
    ##Input: a numpy array of size nsamples x nregions x nregions representing an identity matrix for each patient
    #### read the diagnosis from CDR, turn into dict
    diag = pd.read_csv("ADNI1_patient_diagnosis.csv").set_index('RID')
    diag = diag.astype(float)
    #print(diag)

    ## head count for each category
    # print(diag['m48'].value_counts())

    labels = {}
    for i in diag.index.values:
        if diag.loc[i,'m48'] == 0:
            labels[i] = "Normal"
        elif diag.loc[i,'m48'] == 0.5:
            labels[i] = "Questionable"
        elif diag.loc[i,'m48'] == 1:
            labels[i] = "Mild"
        elif diag.loc[i,'m48'] == 2:
            labels[i] = "Moderate"
        elif diag.loc[i,'m48'] == 3:
            labels[i] = "Severe"
        else:
            labels[i] = 'Not determined'
    #print(labels)

    normal_group = []
    AD_group = []
    combined_group = []

    # print(labels.keys())

    corr_mats = os.listdir('Longitudinal_FreeSurfer_adjmat')
    # patient_dict = {}
    # pid_arr = []

    As = np.zeros((len(corr_mats),116,116))
    count = 0

    for i in corr_mats:
    #     df = pd.read_csv('CrossSectional_FreeSurfer_adjmat/' + i, index_col=0)
        df = pd.read_csv('Longitudinal_FreeSurfer_adjmat/' + i, index_col=0)
        pat = int(i.split('.')[0].split('_')[1])
            
        if df.shape[0] == 116:
            if labels[pat] == 'Normal' or labels[pat] == 'Questionable':
                As[count] = df
                count += 1
                
                normal_group = np.append(normal_group, pat)
                combined_group = np.append(combined_group, pat)
            elif labels[pat] == 'Mild' or labels[pat] == 'Moderate' or labels[pat] == 'Severe':
                As[count] = df
                count += 1
                
                AD_group = np.append(AD_group, pat)
                combined_group = np.append(combined_group, pat)
            else:
                As = np.delete(As, -1, axis=0)
        else:
            As = np.delete(As, -1, axis=0)

    # print(count)
    print(np.shape(As))
    print(np.size(combined_group))

    N = count
    M = 116

    l = pd.read_csv('l_494by4.csv', sep=',',header=None).values
    h = pd.read_csv('h_116by4.csv', sep=',',header=None).values

    ###simulation 10,000 to find 95 percentile
    real_max_byAll, ratio_byAll = find_max_ratio_byAll(normal_group)
    print(real_max_byAll)
    print(ratio_byAll.shape)

    np.savetxt('jointgraph_ratio_byAll.csv', ratio_byAll, delimiter=',')

    max_byAll_arr = np.zeros(10000)
    temp_arr3 = np.zeros((116,4))
    for i in range(10000):
        simulated_normal = np.random.choice(combined_group, len(normal_group), replace=False)
        #simulated_normal = np.setdiff1d(combined_group, AD_group)
        
        max_byAll_arr[i], temp_arr3 = find_max_ratio_byAll(simulated_normal)
        
        if i%1000 == 0:
            print(str(i) + " done")

    np.savetxt('joingraph_sim_max_byAll10000.csv', max_byAll_arr, delimiter=',')
    byAll_cutoff_threshold = np.percentile(max_byAll_arr, 95)
    print(byAll_cutoff_threshold)

    ### find significant structure and dim combo
    for r in range(116):
        for c in range(4):
            if ratio_byAll[r,c] > byAll_cutoff_threshold:
                print("we reject the null for structure " + str(r) + " dim " + str(c) + 
                      " with ratio " + str(ratio_byAll[r,c]))

    np.savetxt('jointgraph_byAll_cutoff.csv', byAll_cutoff_threshold, delimiter=',')
