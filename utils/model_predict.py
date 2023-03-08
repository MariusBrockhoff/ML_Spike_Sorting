# -*- coding: utf-8 -*-
import numpy as np


def model_predict_latents(model, dataset, dataset_test):
    i=0
    for step, batch in enumerate(dataset):
        batch_s = batch[0]
        y_true = y_true + batch[1].numpy().tolist()
        [encoded, latents, output] = model(batch_s)
        if i == 0:
            encoded_data = latents
        encoded_data = np.concatenate((encoded_data, latents))
        i += 1

    i=0
    for step, batch in enumerate(dataset_test):
        batch_s = batch[0]
        y_true = y_true + batch[1].numpy().tolist()
        [encoded, latents, output] = model(batch_s)
        if i == 0:
            encoded_data_test = latents
        encoded_data_test = np.concatenate((encoded_data_test, latents))
        i += 1

    return encoded_data, encoded_data_test
