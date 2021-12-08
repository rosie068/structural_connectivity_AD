# structural_connectivity_AD

This is the repository where I will saving my work on studying correlations between brain structural volumes over time in patients with Alzheimer's vs. healthy.

Assessment.ipynb: processing CDR ratings on all patients in the study from ADNI1 dataset.

Join graph embedding - Daniel.ipynb: estimates the l(#patients by #dimensions) and h(#structures by #dimensions) matrices from patient data using SVDs. Then from the estimate matrices, identify significant networks(dimensions), structures and network-structure pairs based on F-type testing.

Mass_univariate_method_baseline.ipynb: baseline to test for group differences using univariate method: running regression on each structure/patient to get atrophy rate, then testing for difference between the atrophy rates between groups. Test for group difference.

adjacency_mat.ipynb: construct adjacency matrices from correlation matrices for every patient.

corr_mat.ipynb: construct correlation matrices between structures based on volume of structure over 3 or more visits every patient

data_simulation.ipynb: simulate data using fixed l and variable l

joint_graph_daniel.py: python script for running l and h estimation

omnibus_embessing.py: omnibus model where we run SVD on one matrix(diagonal blocks are patient adj matrix, off-diagonal block i,j = (adj matrix for patient i + patient j)/2. Test for group difference.
omnibus_embessing.sh: shell script to run on Hoffman2
