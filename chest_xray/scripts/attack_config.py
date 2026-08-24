"""Shared adversarial attack hyperparameters."""

DEFAULT_STEPS = 10
DEFAULT_MU = 1.0

# Fine-grained L_inf budget for comparing PGD vs BIM before saturation.
# 1/255–6/255: differentiation zone; 8/255: near-total collapse anchor.
EPSILONS = [i / 255 for i in [1, 2, 3, 4, 5, 6, 8]]
