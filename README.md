# Conflicting bundles
In the original source we used much more different architectures and models, 
therefore we provide a re-implementation of "Conflicting bundles" 
here which keeps the code simple, understandable and as a starting point 
for new research. We keep the multi GPU setup (single GPU also works) but 
removed the multi node training for code simplicity.

Note: The vectorized measurement of the bundle entropy is implemented 
in ```conflicting_bundle.py```.


# Dependencies
For this installation we assume that python3, pip3 and all nvidia drivers
(GPU support) are already installed. Then execute the following
to create a virtual environment and install all necessary dependencies:

1. Create virtual environment: ```python3 -m venv env```
2. Activate venv: ```source env/bin/activate```
3. Update your pip installation: ```pip3 install --upgrade pip```
4. Install all requirements. Use requirements-gpu if a gpu is available, requirements-cpu otherwise: ```pip3 install -r requirements.txt```

Note: If the dataset is not available in your home dir, it will be downloaded 
automatically and can take a few minutes.


# Training
Note: A checkpoint is created after each epoch to be able to evaluate the model 
and to evaluate the bundle entropy. To execute all experiments 
about 900GB of storage are needed. You can see an example how to train and 
evaluate models below or you can also execute the run.sh script.

## Network without residual connections
To train and evaluate networks without residuals the following code can be used.
For other datasets or depths the parameters num_layers and dataset must be 
changed accordingly.

```bash
...
python3 train.py --dataset="cifar" --num_layers="50" --log_dir="no_residuals/50" --use_residual="False"
python3 train.py --dataset="cifar" --num_layers="76" --log_dir="no_residuals/76" --use_residual="False"
...
```

## Network with residual connections
To train and evaluate networks with residuals the following code can be used.
For other datasets or depths the parameters num_layers and dataset must be 
changed accordingly.

```bash
...
python3 train.py --dataset="cifar" --num_layers="50" --log_dir="residuals/50" --use_residual="True"
python3 train.py --dataset="cifar" --num_layers="76" --log_dir="residuals/76" --use_residual="True"
...
```

## Auto-tune
The following lines should be executed to train and evaluate an automatically 
pruned model using the auto-tune algorithm proposed in the paper:

```bash
python3 auto_tune.py --dataset="cifar" --log_dir="auto_tune/0"
python3 auto_tune.py --dataset="cifar" --log_dir="auto_tune/1"
python3 auto_tune.py --dataset="cifar" --log_dir="auto_tune/2"
```

# Evaluation
To evaluate a model trained with fixed architecture setups or for a model 
pruned with auto-tune, the ```evaluate.py``` script can be used. Results are 
written into a csv files to e.g. create graphs automatically or also 
tensorboard can be used to show all results (gradients, bundle entropy etc.).

```bash
python3 evaluate.py --dataset="cifar" --log_dir="no_residuals/50" 
python3 evaluate.py --dataset="cifar" --log_dir="no_residuals/76" 
...
python3 evaluate.py --dataset="cifar" --log_dir="residuals/50" 
python3 evaluate.py --dataset="cifar" --log_dir="residuals/76" 
...
python3 evaluate.py --dataset="cifar" --log_dir="auto_tune/0"
python3 evaluate.py --dataset="cifar" --log_dir="auto_tune/1"
python3 evaluate.py --dataset="cifar" --log_dir="auto_tune/2"
```