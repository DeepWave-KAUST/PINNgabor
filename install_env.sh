#!/bin/bash
# 
# Installer for package
# 
# Run: ./install_env.sh
# 

echo 'Creating Package environment'

# create conda env
conda env create -f pinnhash.yml
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda activate pinnhash
conda env list
echo 'Created and activated environment:' $(which python)

echo 'Done!'

