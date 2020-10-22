# PopSkipJumpAttack
Decision Based Adversarial Attack for Noisy classifiers 

## Install Dependencies
This repository is compatible with python 3.6.9 and above.

```pip install -r requirements.txt```

## How to Run

To Run PopSkipJump using default hyperparameters, simply run following command
```python app.py -d {dataset} -n {noise} -a {attack}```

For example, one can run PSJ on deterministic classifier for MNSIT as follows
```python app.py -d mnist -n deterministic -a psj```

For help regarding arguments
```python app.py -h```

## Changing Configurations
You can adjust various settings of the attack in `defaultparams.py` present in root directory of the repository.
For example, you can adjust number of iterations, sampling frequencies and so on. 