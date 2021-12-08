#!/bin/bash

. /u/local/Modules/default/init/modules.sh
module load anaconda3
pip install pandas --user
pip install scipy --user

python3 joint_graph_daniel.py ##omnibus_embedding.py