import time
import numpy as np
import random


def current_time_millis():
    return int(round(time.time() * 1000))

def str_time(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def set_seed(seed:int=None):
    if seed is None:
        pass
    else:
        print("🌱 Setting seed to {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
    return 