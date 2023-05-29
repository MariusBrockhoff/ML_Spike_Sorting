# -*- coding: utf-8 -*-
import numpy as np

def model_predict_latents(model, dataset, dataset_test):
    i = 0
    for step, batch in enumerate(dataset):
        batch_s = batch[0]
        [encoded, latents, output] = model(batch_s)
        if i == 0:
            encoded_data = latents
            y_true = batch[1].numpy().tolist()
        encoded_data = np.concatenate((encoded_data, latents))
        y_true = y_true + batch[1].numpy().tolist()
        i += 1

    i = 0
    for step, batch in enumerate(dataset_test):
        batch_s = batch[0]
        [encoded, latents, output] = model(batch_s)
        if i == 0:
            encoded_data_test = latents
            y_true_test = batch[1].numpy().tolist()
        encoded_data_test = np.concatenate((encoded_data_test, latents))
        y_true_test = y_true_test + batch[1].numpy().tolist()
        i += 1

    return encoded_data, encoded_data_test, np.array(y_true), np.array(y_true_test)


def model_predict_latents_DINO(model, dataset, dataset_test):
    m_student = model[0]
    m_teacher = model[1]
    i = 0
    for step, batch in enumerate(dataset):
        batch_s = batch[0]
        _, student_logits = m_student(batch_s)
        _, teacher_logits = m_teacher(batch_s)
        if i == 0:
            encoded_data_s = student_logits
            encoded_data_t = teacher_logits
            y_true = batch[1].numpy().tolist()
        encoded_data_s = np.concatenate((encoded_data_s, student_logits))
        encoded_data_t = np.concatenate((encoded_data_t, teacher_logits))
        y_true = y_true + batch[1].numpy().tolist()
        i += 1
    encoded_data = np.concatenate((encoded_data_s, encoded_data_t), axis=0)
    i = 0
    for step, batch in enumerate(dataset_test):
        batch_s = batch[0]
        _, student_logits = m_student(batch_s)
        _, teacher_logits = m_teacher(batch_s)
        if i == 0:
            encoded_data_test_s = student_logits
            encoded_data_test_t = teacher_logits
            y_true_test = batch[1].numpy().tolist()
        encoded_data_test_s = np.concatenate((encoded_data_test_s, student_logits))
        encoded_data_test_t = np.concatenate((encoded_data_test_t, teacher_logits))
        y_true_test = y_true_test + batch[1].numpy().tolist()
        i += 1
    encoded_data_test = np.concatenate((encoded_data_test_s, encoded_data_test_t), axis=0)
    return encoded_data, encoded_data_test, np.array(y_true), np.array(y_true_test)