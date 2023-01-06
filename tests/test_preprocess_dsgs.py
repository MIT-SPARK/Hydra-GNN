from hydra_gnn.preprocess_dsgs import _hydra_object_feature_converter
import numpy as np
import gensim
import pandas as pd
import warnings
import os.path

tol = 1e-5

def test_hydra_object_feature_converter(project_data_dir, tol=tol):
    # read hydra object index to color/semantic label conversion file and load word2vec model
    colormap_data_path = project_data_dir / "colormap.csv"
    word2vec_model_path = project_data_dir / "GoogleNews-vectors-negative300.bin"
    if not os.path.exists(colormap_data_path):
        warnings.warn(UserWarning("colormap.csv not found. --skip test"))
        return
    if not os.path.exists(word2vec_model_path):
        warnings.warn(UserWarning("GoogleNews-vectors-negative300.bin word2vec model not found. -- skip test"))
        return 

    colormap_data = colormap_data = pd.read_csv(colormap_data_path, delimiter=',')
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)

    feature_converter = _hydra_object_feature_converter(colormap_data, word2vec_model)
    assert(np.linalg.norm(feature_converter(3) - word2vec_model['chair']) < tol)
    assert(np.linalg.norm(feature_converter(13) - (word2vec_model['chest'] + word2vec_model['drawers'])/2) < tol)
    assert(np.linalg.norm(feature_converter(22) - (word2vec_model['tv'] + word2vec_model['monitor'])/2) < tol)
    assert(np.linalg.norm(feature_converter(36) - word2vec_model['clothes']) < tol)
