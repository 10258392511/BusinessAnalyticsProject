import numpy as np

from sklearn.linear_model import Lasso


CONFIG = {
    "model_cls": Lasso,
    "params": {
        "alpha": 1e-10,
    },
    "param_space": {
        "alpha": [1e-10, 1e-8, 1e-6, 1e-5,1e-4,1e-3,1e-2],
        
    }
}

