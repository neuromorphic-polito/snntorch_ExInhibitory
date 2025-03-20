# snntorch_network


## Dataset analysis

The notebook `notebook/KDE_metrics.ipynb` performs an analysis of the separability of the classes in the WISDM dataset.

## NNI experiments

### Hierarchy

 - For each nni folder, you can run multiple experiments.
 - In an nni experiment, there are n trials.
 - In each trial, we call the python script with one of the combinations in the search space.
 - For each trial, we get the max accuracy for nni to understand how to move to the next space point.


### Creating the experiment(s)

A new experiment can be created with the script

```bash
scripts/gen_new_experiment.sh <experiment_name>

```

This script will create template files in the `nni_experiments/<experiment_name>` folder.

The structure of this folder will be as follows:

```bash
nni_experiments/<experiment_name>
├── code
│   ├── config.py
│   └── experiment1.py
├── config
│   └── config_1.yml
├── results
└── search_space
    └── search_space1.json

```

 - `config_1.yaml` contains nni configurations. It is not necessary to modify this file.
- `search_space1.json` contains the search space for the experiment. This file should be modified to define the search space.
- `config.py` contains configurations that will be common to all the nni experiments. It can be used to modify the trials without changing the code, for example the epochs.
- `experiment1.py` contains the code that will be executed for each trial.
- `results` is the folder where the results of the experiment will be stored.

### Running the experiment

To run the experiment, we use the nni cli:

```bash
nnictl create --config nni_experiments/<experiment_name>/config/config_1.yml -p <port>
```

### Monitoring the experiment

The monitoring can be done with the nni web UI. To access it, open a browser and go to `http://localhost:<port>`


### Visualizing results

The notebook `notebooks/experiment_visualizer.ipynb` can be used to visualize the results of the experiment.

### Existing experiments

Subsets are found using a weighted sum between the **mse** and the **sum**. Different subsets use different weights.

 - **Inhibitory_lif_no_encoder** the second in the list using sum = 1, mse = 0. Chosen because looking at the distance matrix, it has better separability (by eye). (The first in the list is the same as sum = 1, mse = 1).
 - **Inhibitory_lif_no_encoder_balanced** sum =1, mse = 2
 - **Inhibitory_lif_no_encoder_balanced_no_net_loss** sum = 1, mse = 2, without activation sparsity loss
 - **Inhibitory_lif_no_encoder_best** sum = 1, mse = 1
 - **Inhibitory_lif_no_encoder_worst** The last in the list using sum = 1.
 - **Inhibitory_lif_no_encoder_subset2** Custom subset


## Scripts

The docker image contains all dependencies to run the project.
 - `build.sh` builds the docker image.
 - `Dockerfile` contains the docker image configuration.
 - `run.sh` runs the docker image and opens the cli.
 - `run_experiment.sh` runs the experiment in the docker image.
 - `jupyter_server.sh` runs the server in the docker image. You can connect from outside (like from vscode) to the server.
 - `gen_new_experiment.sh` generates a new experiment folder.
