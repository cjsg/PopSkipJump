# Set this to true if you want logits to be treated as probability distribution
NOISE = 'bayesian'  # BOOLEAN

# Number of samples needed to make prediction from non-deterministic model
SAMPLING_FREQ = 20  # INT

# Confidence threshold for sampling from non-deterministic models
SAMPLING_CONF = 0.55  # FLOAT between 0 and 1

# Number of Iterations of the Attack
NUM_ITERATIONS = 64  # INT (Default is 64)

# Path of the image that needs to be attacked
ATTACK_INPUT_IMAGE = None  # STRING
# Label of the image that needs to be attacked
ATTACK_INPUT_LABEL = None  # INT

# Path of initial seed image from where the attack will begin
# If None, The attack will start from a random perturbation
# ATTACK_INITIALISE_IMAGE = 'data/mnist_05_9.jpg'  # STRING
ATTACK_INITIALISE_IMAGE = None  # STRING

# Set this to true if you want Human to act as a model
# Note that for humans, parameter `BAYESIAN` is redundant
ASK_HUMAN = False  # BOOLEAN

EXPERIMENT = True

SLACK = 0.00
FLIP_PROB = 0.20
