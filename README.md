# PopSkipJumpAttack
Decision Based Adversarial Attack for Noisy classifiers 

## Install Dependencies
This repository is compatible with python 3.6.9 and above.

```pip install -r requirements.txt```

## How to Run

To Run PopSkipJump using default hyperparameters, simply run following command

```python app.py -d {dataset} -n {noise} -a {attack} -o {experiment_name}```

For example, one can run PSJ on deterministic classifier for MNSIT as follows

```python app.py -d mnist -n deterministic -a psj -o first_exp```

For help regarding arguments

```python app.py -h```

## Analysing Attack
One can track the progression of attack using the following command

```python plot_distance.py {experiment_name}```

This will output the median l2 distance as a function of iterations of the attack. 

## Plotting Expermiments
`plot_paper.py` can be used to plot the experimental results presented in the paper. 
However one needs to perform all the required experiments to be able to plot them. 
Nevertheless, we include the code used for plotting in `plot_paper.py`  

## Changing Configurations
You can adjust various settings of the attack in `defaultparams.py` present in root directory of the repository.
For example, you can adjust number of iterations, sampling frequencies and so on. 