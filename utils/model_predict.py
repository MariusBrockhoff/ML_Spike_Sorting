# -*- coding: utf-8 -*-
import numpy as np

def model_predict_latents(model, dataset, dataset_test):
    i = 0
    for step, batch in enumerate(dataset):
        batch_s = batch[0]
        [latents, output] = model(batch_s)
        if i == 0:
            encoded_data = latents
            y_true = batch[1].numpy().tolist()
        encoded_data = np.concatenate((encoded_data, latents))
        y_true = y_true + batch[1].numpy().tolist()
        i += 1

    i = 0
    for step, batch in enumerate(dataset_test):
        batch_s = batch[0]
        [latents, output] = model(batch_s)
        if i == 0:
            encoded_data_test = latents
            y_true_test = batch[1].numpy().tolist()
        encoded_data_test = np.concatenate((encoded_data_test, latents))
        y_true_test = y_true_test + batch[1].numpy().tolist()
        i += 1

    return encoded_data, encoded_data_test, np.array(y_true), np.array(y_true_test)
