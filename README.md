# cluster_startup
Getting started on the UChicago DSI cluster


## Outline

### Setting Up Conda on the Cluster

You will need to install your own version of miniconda on the cluster. [Here is a link to the instructions.](https://github.com/uchicago-dsi/core-facility-docs/blob/main/slurm.md#part-vi-install-conda-for-environment-management)


To create the environment from scratch, here are the commands I ran:
```
srun -p general --gres=gpu:1 --pty bash
conda create -n cluster_startup python=3.10
conda activate cluster_startup
conda install pytorch pytorch-cuda -c pytorch -c nvidia
conda install scipy
conda env export --from-history > env.yaml
```

To make sure the installation of conda gives GPU access:
```
python -c "import torch; print(torch.cuda.is_available())"
```

### Requesting a job with a GPU

To request an interactive job with a GPU on the DSI cluster, you can run the following command:
```
srun -p general --gres=gpu:1 --pty bash
```
Other optional flags that might be helpful: `--mem=200G` for 200GB of memory, or `--nodelist=g002` if you only want jobs to run on node g002.

### Training a neural network

Once you are in an interactive job, you can run the model training script with the following command:

```
python train_network.py \
-n_epochs 20 \
--debug \
-train_results_dir results \
-models_dir models
```

After running the python script, try submitting a longer-running batch job. First, you need to edit some lines in the file `long_model_training_script.sh`. Once the file is edited, you can submit the job with:

```
sbatch long_model_training_script.sh
```

### Connecting the server to a Jupyter notebook

### Practical tricks: tmux, squeue, saving outputs
