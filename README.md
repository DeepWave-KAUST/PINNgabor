[LOGO](https://github.com/DeepWave-Kaust/PINNgabor-dev/blob/main/asset/diagram.png)

Reproducible material for GaborPINN: Efficient physics informed neural networks using multiplicative filtered networks - **Xinquan Huang, Tariq Alkhalifah.**

# Project structure
This repository is organized as follows:

* :open_file_folder: **pinngabor**: python library containing the main code and utils;
* :open_file_folder: **asset**: folder containing logo;
* :open_file_folder: **data**: folder containing data;
* :open_file_folder: **scripts**: set of python scripts used to run multiple experiments.

## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `pinnhash.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. 

Remember to always activate the environment by typing:
```
conda activate pinnhash
```

## Scripts
Go to folder `scripts` and run
```
bash run.sh
```
After running, go to folder `exp/results/tb` in the root_path produced by the procedures, and you could use `tensorboard` to visualize the trainig process and predictions.

#### Change the root_path
In the `run.sh` script, you need to modify the variable from line 4 to 6 (`tb_root, run_root, data_root`) to specify the root path for your procedures.

#### Check the results
After finish the training, you could go to the `<run_root>/results/tb` to use `tensorboard --logdir=./` to check the training metrics and testing results.

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce A6000 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.

## Cite us 
```bibtex
@article{huang2023gaborpinn,
  title={GaborPINN: Efficient physics informed neural networks using multiplicative filtered},
  author={Huang, Xinquan and Alkhalifah, Tariq},
  journal={IEEE Geoscience and Remote Sensing Letters},
  volume={20},
  pages={1--5},
  year={2023},
  doi={10.1109/LGRS.2023.3330774},
  publisher={IEEE}
}
```

## Acknowledgement
This code is developed based on open-sourced projects [Multiplicative-filter-networks](https://github.com/boschresearch/multiplicative-filter-networks).
