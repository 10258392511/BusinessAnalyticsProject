import numpy as np

from sklearn.linear_model import Lasso


CONFIG = {
    "model_cls": Lasso,
    "params": {
        "alpha": 1e-3,
        "tol": 1e-4
    },
    "param_space": {
        "alpha": [1e-10, 1e-8, 1e-6, 1e-5,1e-4],
        "tol": [1e-3,5e-3,1e-2,5e-2,1e-1,5e-1,1,5,10]
        
    }
}
