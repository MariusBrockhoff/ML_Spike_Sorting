import numpy as np


def model_predict_latents(model, dataset, dataset_test):
    """
    Generate latent representations from a given model using training and testing datasets.

    This function passes batches of data through the model to obtain latent representations. It concatenates
    these representations for all batches in the training and testing datasets. It also collects the true labels
    from the batches.

    Arguments:
        model: A machine learning model that has a method to generate latent representations.
        dataset: An iterable (like a DataLoader) that provides batches of training data.
                 Each batch is expected to be a tuple where the first element is the input data
                 and the second element is the corresponding labels.
        dataset_test: Similar to `dataset`, but for testing data.

    Returns:
        encoded_data: A numpy array containing the concatenated latent representations from the training data.
        encoded_data_test: A numpy array containing the concatenated latent representations from the testing data.
        y_true: A numpy array of the true labels from the training data.
        y_true_test: A numpy array of the true labels from the testing data.
    """
    i = 0
    for step, batch in enumerate(dataset):
        batch_s = batch[0]
        [latents, _] = model(batch_s)
        if i == 0:
            encoded_data = latents
            y_true = batch[1][:,0].numpy().tolist()
        encoded_data = np.concatenate((encoded_data, latents))
        y_true = y_true + batch[1][:,0].numpy().tolist()
        i += 1

    i_test = 0
    for step_test, batch_test in enumerate(dataset_test):
        batch_s = batch_test[0]
        [latents, _] = model(batch_s)
        if i_test == 0:
            encoded_data_test = latents
            y_true_test = batch_test[1][:,0].numpy().tolist()
        encoded_data_test = np.concatenate((encoded_data_test, latents))
        y_true_test = y_true_test + batch_test[1][:,0].numpy().tolist()
        i_test += 1

    return encoded_data, encoded_data_test, np.array(y_true), np.array(y_true_test)