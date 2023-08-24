# cluster_startup
Getting started on the UChicago DSI cluster


## Outline

### How to set up Git on cluster?
1. How can we save git user information on the cluster? 
```
git config --global --unset credential.helper
git config --global credential.helper store
```
2. Please see the stepwise screencuts here. [Like to set up your password](https://docs.google.com/document/d/13S4rIdJCzNqi_myG9TcjPyorI2n9IHbaqvS8Ao52R5Q/edit?usp=sharing)
3. [Git cheat sheet.](https://doabledanny.gumroad.com/l/git-commands-cheat-sheet-pdf)

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

### Know your slurm cluster.
```
squeue % Shows the state of jobs.
scancel 12345 % Used to cancel a job.
sprio % Displays the priority of pending jobs.
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

1. Allocate a gpu/cpus.
2. Start your Jupyter notebook in your terminal.
```
bash start_jupyter.sh
```
3. Check your job output. Create an SSH tunnel to forward the notebook interface to your local.
```
ssh -N -L 8888:$NODEIP:$NODEPORT user@fe01.ai.cs.uchicago.edu
```
4. Open your local browser and visit: http://localhost:8888.

### Practical tricks: tmux, squeue, saving outputs
1. Why do we need to use tmux? tmux is a terminal multiplexer. This means it allows multiple terminal sessions to be created, accessed, and controlled from a single screen. tmux may be detached from a screen and *continue running in the background*, then later reattached.
```
tmux % Create a new session
tmux ls % List running sessions
```
For window management:
```
Window Management:

Ctrl-b c: Create a new window
Ctrl-b %: Split the window vertically into panes
Ctrl-b ": Split the window horizontally into panes
Ctrl-b n: Go to the next window
Ctrl-b p: Go to the previous window
Ctrl-b x: Kill the current window
Ctrl-b fn-arrowup/arrowdown: scroll in your window
```

