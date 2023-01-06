from hydra_gnn.preprocess_dsgs import _hydra_object_feature_converter
import numpy as np
import gensim
import pandas as pd
import warnings
import pytest

tol = 1e-5

def test_hydra_object_feature_converter(tol=tol):
    if pytest.colormap_data is None or pytest.word2vec_model is None:
        warnings.warn(UserWarning("data file(s) missing. -- skip test"))
        return
    else:
        colormap_data = pytest.colormap_data
        word2vec_model = pytest.word2vec_model

    # feature converter should convert hydra integer label to the corresponding 300-dim word2vec feature vector
    feature_converter = _hydra_object_feature_converter(colormap_data, word2vec_model)
    assert(np.linalg.norm(feature_converter(3) - word2vec_model['chair']) < tol)
    assert(np.linalg.norm(feature_converter(13) - (word2vec_model['chest'] + word2vec_model['drawers'])/2) < tol)
    assert(np.linalg.norm(feature_converter(22) - (word2vec_model['tv'] + word2vec_model['monitor'])/2) < tol)
    assert(np.linalg.norm(feature_converter(36) - word2vec_model['clothes']) < tol)
