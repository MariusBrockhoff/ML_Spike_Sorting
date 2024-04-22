# PseudoSorter: Machine learning based Spike Sorting

## Prerequisites
You will need the following to run our code:
* Python 3
* Conda
* Weights and Biases (recommended, optional)

## Getting started
### Launch the virtual environment
1. We recommend launching a virtual environment to install all the necessary Python packages. Multiple options are available, when using conda, use:

    `conda create --name <env_name> --file requirements.txt`
2. To launch the created virtual environment that contains all the necessary Python packages, run
`conda activate env_name` from the root directory. 
The virtual environment should be active. Make sure the correct versions of CUDA and cuDNN are installed.

## Run PseudoSorter
### For Benchmarking on simulated spike recordings
Two distinct usages for PseudoSorter are available. Firstly, for benchmarking of simulated spike shape recordings, PseudoSorter runs in 'benchmark=True' mode.
Here, it's assumed that Ground truth labels for each spike are available. Input are Python .pkl files with the isolated spike shape recordings (see publication below for details on the structure).
The full PseudoSorter pipeline can be run on a sample dataset at path `sample_path` by calling:
`python run.py --Pretrain_Method NNCLR --Finetune_Method PseudoLabel --Model DenseAutoencoder --PathData sample_path --Benchmark`.
Models and results are saved after pre-training. Therefore, pre-training or fine-tuning can also be called separately by running e.g.
`python run.py --Pretrain_Method NNCLR --Model DenseAutoencoder --PathData sample_path --Benchmark` or 
`python run.py --Finetune_Method PseudoLabel --Model DenseAutoencoder --PathData sample_path --Benchmark`.

### For new, raw MEA recording data
Here, PseudoSorter shall be applied to classify all spikes from a raw MEA recording. Currently, we only support Multi Channel system GmbH acquisition systems (stored as .h5 files). The Raw recordings are filtered, spikes detected and pre-processed and subsequently all sorted at once (aim for high-level sorting).
For a sample raw recording file at path `sample_raw`, run:
`python run.py --Pretrain_Method NNCLR --Finetune_Method PseudoLabel --Model DenseAutoencoder --PathData sample_raw`.


### Important Notes
All parameters can be adjusted in the configuration files. There are separate files for pre-training and fine-tuning. The config_file for data_preprocessing is only relevant when not benchmarking. By default, all training is tracked with Weight and Biases. If you do not wish to track the training, add `--wand False` when running PseudoSorter.


## Acknowledgements
If you use PseudoSorter, please cite our [preprint](https://doi.org/10.1101/2024.02.29.582792 ):

Brockhoff, M., Träuble, J., Middya, S., Fuchsberger, T., Fernandez-Villegas, A., Stephens, A. D., ... & Schierle, G. S. K. (2024). Machine learning-based spike sorting reveals how subneuronal concentrations of monomeric Tau cause a loss in excitatory postsynaptic currents in hippocampal neurons. bioRxiv, 2024-02. DOI: https://doi.org/10.1101/2024.02.29.582792


