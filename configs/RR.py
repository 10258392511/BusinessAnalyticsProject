import numpy as np

from sklearn.linear_model import Ridge


CONFIG = {
    "model_cls": Ridge,
    "params": {
        "alpha": 1,
    },

    "param_space": {
        "alpha": [1e-15, 1e-12, 1e-10,1e-5, 0.001, 0.01, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 1],
    }
}

