import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def read_CDR(CDR_file, adj_path):
    diag = pd.read_csv(CDR_file).set_index('RID')
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

    adj_mats = os.listdir(adj_path)
    As = np.zeros((len(adj_mats),108,108))
    count = 0
    structures = []

    for i in adj_mats:
        df = pd.read_csv(adj_path + '/' + i, index_col=0)
        pat = int(i.split('.')[0].split('_')[1])

        if df.shape[0] == 108:
            if len(structures) == 0:
                structures = df.columns.values

            if labels[pat] == 'Normal' or labels[pat] == 'Questionable':
                As[count] = df
                count += 1
                normal_group = np.append(normal_group, pat)
                combined_group = np.append(combined_group, pat)
            elif (labels[pat] == 'Mild' or labels[pat] == 'Moderate' or 
                  labels[pat] == 'Severe'):
                As[count] = df
                count += 1
                AD_group = np.append(AD_group, pat)
                combined_group = np.append(combined_group, pat)
            else:
                As = np.delete(As, -1, axis=0)
        else:
            As = np.delete(As, -1, axis=0)

    print(np.shape(As))
    print(np.size(combined_group))

    N = count
    M = len(structures)
    
    return As, N, M, normal_group, AD_group, combined_group


def joint_graph_embedding(el, e, d, As, N, M):
    ## Setup
    l = torch.randn(N,d,requires_grad=True)
    h = torch.randn(M,d)
    h /= torch.sqrt(torch.sum(h**2,0,keepdims=True))
    h.requires_grad = True

    At = torch.tensor(As,dtype=h.dtype)
    eiginit = False
    if eiginit:
        h = torch.symeig(torch.mean(At,0),eigenvectors=True)[1]
        h = h.flip(1)
        h = h[:,:d]
        h.requires_grad = True
        with torch.no_grad():
            hh_ = (h[:,None]*h).reshape(-1,d)
            At_ = At.reshape(At.shape[0],-1)            
            l = torch.solve((At_@hh_).T, (hh_.T@hh_) )[0].T                      

        Ahat = (l[:,None,:]*h)@h.T
        print(f'err = {torch.sum((At-Ahat)**2)*0.5}')
        f,ax = plt.subplots()
        im = ax.imshow(torch.mean(Ahat,0).detach().cpu().numpy())
        plt.colorbar(im)

        l.requires_grad = True

    #leye = torch.zeros(N,1,dtype=h.dtype,requires_grad=True)
    #eye = torch.eye(M,dtype=h.dtype)

    ## Estimation
    f,ax = plt.subplots()
    f2,ax2 = plt.subplots()
    Esave = []
    niter = 10000
    n_draw = 500
    for d_ in range(d):
        for it in range(niter):
            hd = h[:,:d_+1]
            ld = l[:,:d_+1]                        

            Ahat0 = (ld[:,None,:]*hd)@hd.T # + eye*leye[:,None,:]
            Ahat = torch.sigmoid(Ahat0)
                        
            ### applying a mask to make estimated At diagonal 0, then it won't be taken into account
            mask = torch.ones([As.shape[1], As.shape[1]])
            mask.fill_diagonal_(0)
            Ahat0 = torch.mul(Ahat0, mask)
            
            E = torch.nn.functional.binary_cross_entropy_with_logits(Ahat0,At,
                                                                     reduction='mean')
            Esave.append(E.item())        


            if not it%n_draw or it==niter-1:
                ax.cla()
                ax.plot(Esave)
                f.canvas.draw()

                f_out = "itr=10000_el=" + str(el) + "_e=" + str(e) + "_f.png"
                f.savefig(f_out)

                f2.clf()
                ax2 = f2.add_subplot()
                im = ax2.imshow(torch.mean(Ahat.clone().detach(),0).cpu().numpy(),
                                vmin=0.0,vmax=1.0)
                plt.colorbar(im)
                f2.canvas.draw()

                f2_out = "itr=10000_el=" + str(el) + "_e=" + str(e) + "_f2.png"
                f2.savefig(f2_out)


            # now update
            E.backward()
            with torch.no_grad():
                update_all = False
                if update_all:
                    hgrad = h.grad[:,:d_+1]*e    
                    hgrad = hgrad - torch.sum(hgrad*hd,0)*hd            
                    h[:,:d_+1] -= hgrad
                else:
                    hgrad = e*h.grad[:,d_]
                    hgrad = hgrad - torch.sum(hgrad*hd[:,-1],0)*hd[:,-1]
                    h[:,d_] -= hgrad

                h /= torch.sqrt(torch.sum(h**2,0,keepdims=True))

                l -= el*l.grad            
                #leye -= el*leye.grad

                h.grad.zero_()
                l.grad.zero_()
                #leye.grad.zero_()
        print(Esave[-1])
        print(ld)

    l = l.detach().numpy()
    h = h.detach().numpy()

    np.savetxt('l.csv', l, delimiter=',')
    np.savetxt('h.csv', h, delimiter=',')
    
    return l, h



def construct_tensor(l,h):
    num_pat = np.shape(l)[0]
    num_struct = np.shape(h)[0]
    num_dim = np.shape(h)[1]

    tensor_all = np.zeros((num_pat, num_dim, num_struct, num_struct))

    for j in range(num_pat):
        for i in range(num_dim):
            l_ij = l[j,i]
            h_jk = np.mat(h[:,i])
            this_dim_mat = np.matmul(np.transpose(h_jk), h_jk) * l_ij
            tensor_all[j,i,:,:] = this_dim_mat
    
    return tensor_all


def find_ratio_by_dim(normal):
    ratio_by_dim = np.zeros(num_dim)
    
    for i in range(num_dim):
        combined_by_dim = np.zeros((len(combined_group), num_struct, num_struct))
        normal_by_dim = np.zeros((len(normal_group), num_struct, num_struct))
        AD_by_dim = np.zeros((len(AD_group), num_struct, num_struct))
        normal_itr = 0
        AD_itr = 0
        
        for j in range(len(combined_group)):
            combined_by_dim[j,:,:] = tensor_all[j,i,:,:]

            p = combined_group[j]        
            if p in normal:
                normal_by_dim[normal_itr,:,:] = tensor_all[j,i,:,:]
                normal_itr += 1
            else:
                AD_by_dim[AD_itr,:,:] = tensor_all[j,i,:,:]
                AD_itr += 1

        combined_group_mean = np.mean(combined_by_dim, axis = 0)
        normal_group_mean = np.mean(normal_by_dim, axis = 0)
        AD_group_mean = np.mean(AD_by_dim, axis = 0)

        combined_SSE = np.sum(np.square(combined_by_dim - combined_group_mean))
        normal_SSE = np.sum(np.square(normal_by_dim - normal_group_mean))
        AD_SSE = np.sum(np.square(AD_by_dim - AD_group_mean))

        ratio = combined_SSE - (normal_SSE+AD_SSE)
        ratio_by_dim[i] = ratio
    
    max_ratio = np.max(ratio_by_dim)
    return max_ratio, ratio_by_dim



def testing_for_networks():
    real_max_byDim, ratio_byDim = find_ratio_by_dim(normal_group)

    max_byDim_arr = np.zeros(10000)
    temp_arr = np.zeros(4)
    for i in range(10000):
        simulated_normal = np.random.choice(combined_group, len(normal_group), 
                                            replace=False)
        max_byDim_arr[i], temp_arr = find_ratio_by_dim(simulated_normal)

        if i%1000 == 0:
            print(str(i) + " done")
    byDim_cutoff_threshold = np.percentile(max_byDim_arr, 95)
    
    np.savetxt('simulated_max_score_by_network.csv', max_byDim_arr, delimiter=',')
    p_val = (max_byDim_arr > real_max_byDim).sum()
    print("p value is " + str(p_val))

    for r in range(len(ratio_byDim)):
        if ratio_byDim[r] > byDim_cutoff_threshold:
            print("we reject the null for dimention " + str(r))



def find_ratio_tuple_triple(normal):
    ratio_by_tuple = np.zeros((num_dim, num_struct))
    ratio_by_triple = np.zeros((num_dim, num_struct, num_struct))
    
    for i in range(num_dim):
        combined_by_dim = np.zeros((len(combined_group), num_struct, num_struct))
        normal_by_dim = np.zeros((len(normal_group), num_struct, num_struct))
        AD_by_dim = np.zeros((len(AD_group), num_struct, num_struct))
        normal_itr = 0
        AD_itr = 0
        
        for j in range(len(combined_group)):
            combined_by_dim[j,:,:] = tensor_all[j,i,:,:]

            p = combined_group[j]        
            if p in normal:
                normal_by_dim[normal_itr,:,:] = tensor_all[j,i,:,:]
                normal_itr += 1
            else:
                AD_by_dim[AD_itr,:,:] = tensor_all[j,i,:,:]
                AD_itr += 1

        combined_group_mean = np.mean(combined_by_dim, axis = 0)
        normal_group_mean = np.mean(normal_by_dim, axis = 0)
        AD_group_mean = np.mean(AD_by_dim, axis = 0)
        
        combined_square_error = np.square(combined_by_dim - combined_group_mean)
        normal_square_error = np.square(normal_by_dim - normal_group_mean)
        AD_square_error = np.square(AD_by_dim - AD_group_mean)
        
        ##### computing tuple, summing over k' and then j
        combined_SSE_tuple = np.sum(combined_square_error, axis=2)
        normal_SSE_tuple = np.sum(normal_square_error, axis=2)
        AD_SSE_tuple = np.sum(AD_square_error, axis=2)
            
        for k in range(num_struct):
            ratio_tuple = np.sum(combined_SSE_tuple[:,k]) - (np.sum(normal_SSE_tuple[:,k])+
                                                             np.sum(AD_SSE_tuple[:,k]))
            ratio_by_tuple[i,k] = ratio_tuple
           
        ##### computing triple, summing over j only
        for l in range(num_struct):
            for m in range(l, num_struct): ##since the matrix is symmetric
                ratio_triple = np.sum(combined_square_error[:,l,m])-(np.sum(normal_square_error[:,l,m])+
                                                                   np.sum(AD_square_error[:,l,m]))
                ratio_by_triple[i,l,m] = ratio_triple
                ratio_by_triple[i,m,l] = ratio_triple
                
    max_ratio_tuple = np.max(ratio_by_tuple)
    max_ratio_triple = np.max(ratio_by_triple)
    return max_ratio_tuple, max_ratio_triple, ratio_by_tuple, ratio_by_triple



##global FWER corrected p value
def testing_for_tuples():
    num_dim = 4
    num_struct = 108
    real_max_tup, real_max_tri, real_ratio_tup, real_ratio_tri = find_ratio_tuple_triple(normal_group)

    max_byTuple_arr = np.zeros(10000)
    max_byTriple_arr = np.zeros(10000)

    temp_arr_tup = np.zeros((num_dim, num_struct))
    temp_arr_tri = np.zeros((num_dim, num_struct, num_struct))
    for i in range(10000):
        simulated_normal = np.random.choice(combined_group, len(normal_group), replace=False)
        max_byTuple_arr[i],max_byTriple_arr[i],temp_arr_tup,temp_arr_tri = find_ratio_tuple_triple(simulated_normal)

        if i%1000 == 0:
            print(str(i) + " done")

    byTuple_cutoff_threshold = np.percentile(max_byTuple_arr, 95)
    byTriple_cutoff_threshold = np.percentile(max_byTriple_arr, 95)
    
    print("******************")
    ##global FWER corrected p value
    tuple_ranked_by_pval = pd.DataFrame(columns=['dimension','structure','p_value'])

    ##local FWER corrected p value
    rej_count = 0
    for i in range(num_dim):
        for k in range(num_struct):
            p_val = (real_ratio_tup[i,k] < max_byTuple_arr).sum() / 10000.0
            if p_val < 0.05:
                tuple_ranked_by_pval.loc[len(tuple_ranked_by_pval.index)] = [i,k,p_val]
                print("we reject the null for dimension " + str(i) +
                      " structure " + str(k) + " with sum " + str(real_ratio_tup[i,k]))
                rej_count+=1
    print("A total of " + str(rej_count) + " network-structure pairs rejected.")
    
    tuple_ranked_by_pval.sort_values(by=['p_value'], inplace=True)
    tuple_ranked_by_pval.to_csv("tuple_ranked_bypval.csv")
    
    print("******************")
    triple_ranked_by_pval = pd.DataFrame(columns=['dimension','structure','structure_2','p_value'])

    rej_count_tri = 0
    for i in range(num_dim):
        for k in range(num_struct):
            for m in range(k, num_struct):
                p_val = (real_ratio_tri[i,k,m] < max_byTriple_arr).sum() / 10000.0
                if p_val < 0.05:
                    triple_ranked_by_pval.loc[len(triple_ranked_by_pval.index)] = [i,k,m,p_val]
                    print("we reject the null for dimension " + str(i) +
                          " structure " + str(k) +
                          " and structure " + str(m) +
                          " with ratio " + str(real_ratio_tri[i,k,m]))
                    rej_count_tri+=1
    print("A total of " + str(rej_count_tri) + " network-structure-structure triplets rejected.")
    triple_ranked_by_pval.sort_values(by=['p_value'], inplace=True)
    triple_ranked_by_pval.to_csv("triple_ranked_bypval.csv")