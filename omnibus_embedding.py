import pandas as pd
import numpy as np
import os
import csv
from scipy import sparse
from scipy.sparse.linalg import eigsh

### finding top eigenvalues of the omnibus matrix
def top_eig(group_arr, num):
    size = 116 * len(group_arr)
    big_mat = sparse.lil_matrix((size,size))

    ### construct sparse matrix with lil, then find eigsh
    for i in range(len(group_arr)):
        pat = group_arr[i]
        pat_mat = patient_dict[pat]
        idx = np.arange(i*116,i*116+116)
        big_mat[i*116:i*116+116,idx] = pat_mat

        for j in range(i+1,len(group_arr)):
            pat2 = group_arr[j]
            avg_mat = (pat_mat + patient_dict[pat2])/2
            #print(avg_mat)

            idx2 = np.arange(j*116,j*116+116)
            big_mat[i*116:i*116+116,idx2] = avg_mat

            idx2_reverse = np.arange(i*116,i*116+116)
            big_mat[j*116:j*116+116,idx2_reverse] = avg_mat
    # print("big_mat finished, number of non-zero entries: " + str(big_mat.count_nonzero()))

    eigenvalues, eigenvectors = eigsh(big_mat, k=num, which='LM')
    return eigenvectors, eigenvalues

### for each large eigenvec, split into patient and find the mean, calculate SSE
def calc_SSE(group, big_eigenvec):
    mean_eigvec = np.mean(big_eigenvec, axis=0)
    
    SSE = 0
    for p in range(len(group)):
        diff = np.subtract(big_eigenvec[p], mean_eigvec)
        SSE += np.sum(np.square(diff))
        
    return SSE

if __name__=="__main__":
    dirname="Longitudinal_FreeSurfer_adjmat"
    parent_dir=os.getcwd()
    path=os.path.join(parent_dir,dirname)
    files=os.listdir(path)

    patient_dict = {}
    pid_arr = []
    for f in files:
        file_p="Longitudinal_FreeSurfer_adjmat/" + f
        mat = pd.read_csv(file_p, index_col=0).to_numpy()
        #print(mat.shape)
        if mat.shape[0] == 116:
            pid = int(f.split('.')[0].split('_')[1])
            pid_arr = np.append(pid_arr, pid)
            patient_dict[pid] = mat

    pid_arr = np.sort(pid_arr)

    #### read the diagnosis from CDR, turn into dict
    diag = pd.read_csv("ADNI1_patient_diagnosis.csv").set_index('RID')
    diag = diag.astype(float)

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

    normal_group = []
    AD_group = []
    combined_group = []

    for pat in pid_arr:
        if labels[pat] == 'Normal' or labels[pat] == 'Questionable':
            normal_group = np.append(normal_group, pat)
            combined_group = np.append(combined_group, pat)
        elif labels[pat] == 'Mild' or labels[pat] == 'Moderate' or labels[pat] == 'Severe':
            AD_group = np.append(AD_group, pat)
            combined_group = np.append(combined_group, pat)

    normal_len = len(normal_group)

    # normal_group = normal_group[:20]
    # AD_group = AD_group[:20]
    # combined_group = np.concatenate((normal_group, AD_group), axis=None)


    '''
    ##testing if i run eigen finder on each individual groups
    ### get top 4 eigenvec for combined group, then split
    combined_vec, combined_val = top_eig(combined_group, 4)
    normal_vec, normal_val = top_eig(normal_group, 4)
    AD_vec, AD_val = top_eig(AD_group, 4)

    #### split combined vec into normal and AD vec
    normal_3d = np.zeros(shape=(len(normal_group),116,4))
    AD_3d = np.zeros(shape=(len(AD_group),116,4))
    combined_3d = np.zeros(shape=(len(combined_group),116,4))

    for i in range(len(combined_group)):
        pat_mat = combined_vec[i*116:i*116+116,:]
        combined_3d[i] = pat_mat
    for j in range(len(normal_group)):
        pat_mat = normal_vec[j*116:j*116+116,:]
        normal_3d[j] = pat_mat
    for k in range(len(AD_group)):
        pat_mat = AD_vec[k*116:k*116+116,:]
        AD_3d[k] = pat_mat

    combined_SSE = calc_SSE(combined_group, combined_3d)
    normal_SSE = calc_SSE(normal_group, normal_3d)
    AD_SSE = calc_SSE(AD_group, AD_3d)

    print("Real combined SSE: " + str(combined_SSE) +
        "\nreal normal SSE: " + str(normal_SSE) + 
        "\nreal AD SSE: " + str(AD_SSE) +
        "\nratio: " + str(combined_SSE / (normal_SSE + AD_SSE)))

    ### permutation testing
    SSE_df = pd.DataFrame(columns=['normal','AD'])
    iteration = 10000
    for itr in range(iteration):
        simulated_normal = np.random.choice(combined_group, normal_len, replace=False)  ## cross sectional there is 30
        simulated_AD = np.setdiff1d(combined_group, simulated_normal)

        #### split combined vec into normal and AD vec
        normal_3d_sim = np.zeros(shape=(len(simulated_normal),116,4))
        AD_3d_sim = np.zeros(shape=(len(simulated_AD),116,4))

        normal_counter = 0
        AD_counter = 0
        for i in range(len(combined_group)):
            pat = combined_group[i]
            pat_mat = combined_vec[i*116:i*116+116,:]

            if pat in simulated_normal:
                normal_3d_sim[normal_counter] = pat_mat
                normal_counter += 1
            else:
                AD_3d_sim[AD_counter] = pat_mat
                AD_counter += 1

        normal_SSE_sim = calc_SSE(simulated_normal, normal_3d_sim)
        AD_SSE_sim = calc_SSE(simulated_AD, AD_3d_sim)
        SSE_arr = [normal_SSE_sim, AD_SSE_sim]
        SSE_df.loc[SSE_df.shape[0]] = SSE_arr

    SSE_df['combined'] = np.full(shape=iteration, fill_value=combined_SSE, dtype=np.float)
    SSE_df['test_ratio'] = SSE_df['combined'] / (SSE_df['normal'] + SSE_df['AD']) 
    SSE_df.to_csv("omnibus_emb_1103.csv", index=False)
    '''


    ##Original code for run once and split
    ### get top 4 eigenvec for combined group, then split
    combined_vec, combined_val = top_eig(combined_group, 4)

    #### split combined vec into normal and AD vec
    normal_3d = np.zeros(shape=(len(normal_group),116,4))
    AD_3d = np.zeros(shape=(len(AD_group),116,4))
    combined_3d = np.zeros(shape=(len(combined_group),116,4))

    normal_counter = 0
    AD_counter = 0
    for i in range(len(combined_group)):
        pat = combined_group[i]
        pat_mat = combined_vec[i*116:i*116+116,:]
        
        combined_3d[i] = pat_mat
        if pat in normal_group:
            normal_3d[normal_counter] = pat_mat
            normal_counter += 1
        else:
            AD_3d[AD_counter] = pat_mat
            AD_counter += 1

    combined_SSE = calc_SSE(combined_group, combined_3d)
    normal_SSE = calc_SSE(normal_group, normal_3d)
    AD_SSE = calc_SSE(AD_group, AD_3d)

    print("Real combined SSE: " + str(combined_SSE) +
        "\nreal normal SSE: " + str(normal_SSE) + 
        "\nreal AD SSE: " + str(AD_SSE) +
        "\nratio: " + str(combined_SSE / (normal_SSE + AD_SSE)))

    ### permutation testing
    SSE_df = pd.DataFrame(columns=['normal','AD'])
    iteration = 10000
    for itr in range(iteration):
        simulated_normal = np.random.choice(combined_group, normal_len, replace=False)  ## cross sectional there is 30
        simulated_AD = np.setdiff1d(combined_group, simulated_normal)

        #### split combined vec into normal and AD vec
        normal_3d_sim = np.zeros(shape=(len(simulated_normal),116,4))
        AD_3d_sim = np.zeros(shape=(len(simulated_AD),116,4))

        normal_counter = 0
        AD_counter = 0
        for i in range(len(combined_group)):
            pat = combined_group[i]
            pat_mat = combined_vec[i*116:i*116+116,:]

            if pat in simulated_normal:
                normal_3d_sim[normal_counter] = pat_mat
                normal_counter += 1
            else:
                AD_3d_sim[AD_counter] = pat_mat
                AD_counter += 1

        normal_SSE_sim = calc_SSE(simulated_normal, normal_3d_sim)
        AD_SSE_sim = calc_SSE(simulated_AD, AD_3d_sim)
        SSE_arr = [normal_SSE_sim, AD_SSE_sim]
        SSE_df.loc[SSE_df.shape[0]] = SSE_arr

    SSE_df['combined'] = np.full(shape=iteration, fill_value=combined_SSE, dtype=np.float)
    SSE_df['test_ratio'] = SSE_df['combined'] / (SSE_df['normal'] + SSE_df['AD']) 
    SSE_df.to_csv("omnibus_emb_1103.csv", index=False)