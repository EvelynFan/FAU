#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import sys
import numpy as np
from datetime import datetime

def approx_equal(a, b, eps=1e-9):
    return np.fabs(a-b) < eps

def get_rng(obj=None):
    """
    Get a good RNG seeded with time, pid and the object.

    Args:
        obj: some object to use to generate random seed.
    Returns:
        np.random.RandomState: the RNG.
    """
    seed = (id(obj) + os.getpid() +
            int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
    return np.random.RandomState(seed)

