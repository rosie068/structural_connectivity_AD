# structural_connectivity_AD

Repository for studying correlations between neuraldegeneration volume over time in patients with mild Alzheimer's vs. severe.

dataset: contains the sample dataset that the paper is based on, taken from ADNI 1 dataset UCSF Longitudinal study

preprocess: notebook on preprocessing data dataset. If input data has the same format as our example, can also preprocess into input format 

pipeline_functions: joint graph embedding algorithm and f-type statistics analysis functions with and without confounder regression, and helper functions

joint_graph_embedding_analysis: notebook on running the algorithm and analysis

requirements.txt: requirement packages and versions

For running on the example dataset, first run preprocess notebook, then run joint_graph_embedding_analysis.

documentation generated with sphinx software (https://www.sphinx-doc.org/en/master/), code documentation can be found in docs/build/html/index.html

