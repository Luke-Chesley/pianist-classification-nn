import re
import torch
import numpy as np
import pickle
import pandas as pd
import string
from contractions import contractions_dict
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import os
import random
import librosa






##########################################################################


def n_most_likely_classes(initial_features, trainer, extractor, id2label, n=3):
    encoding = extractor(initial_features, sampling_rate = 16000, return_tensors="pt", truncation=True)
    encoding = {k: v.to(trainer.model.device) for k, v in encoding.items()}

    outputs = trainer.model(**encoding)

    logits = outputs.logits

    # apply sigmoid + threshold
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(logits.squeeze().cpu())
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.0)] = 1
    # turn predicted id's into actual label names
    predicted_labels = [
        id2label[idx] for idx, label in enumerate(predictions) if label == 1.0
    ]

    pred_label_probs = list(probs[np.where(probs >= 0.0)].detach().numpy().round(3))
        
    if len(predicted_labels) != len(pred_label_probs):
        raise ValueError(
            "The length of 'classes' and 'probabilities' lists must be the same."
        )

    prob_dict = {class_name: prob for class_name, prob in zip(predicted_labels, pred_label_probs)}
    
    sorted_items = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
    top_n_items = sorted_items[:n] # return this for nestted list
    top_n_dict = dict(top_n_items)

    return top_n_dict

##########################################################################
def get_random_files_from_subdirs(parent_dir, num_files=100):
    file_list = []

    for dirpath, _, filenames in os.walk(parent_dir):
        # Shuffle the list of filenames randomly
        random.shuffle(filenames)

        # Take the first 'num_files' files from the shuffled list
        selected_files = filenames[:num_files]

        # Construct full paths for selected files
        selected_paths = [
            os.path.join(dirpath, filename) for filename in selected_files
        ]

        # Extend the file_list with the selected file paths
        file_list.extend(selected_paths)

    return pd.Series(file_list)

##########################################################################
def file_to_librosa_features(file, sample_rate):
    y, sr = librosa.load(file, sr=sample_rate)

    return y


