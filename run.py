# -*- coding: utf-8 -*-
import tensorflow as tf


print("Num GPUs:", len(tf.config.experimental.list_physical_devices(device_type='GPU')))
print("devices:", tf.config.experimental.list_physical_devices(device_type=None))
