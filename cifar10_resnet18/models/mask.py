#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import scipy.sparse as spy
import numpy as np

def quandequan_range(indata, bitlength):
    import torch
    factor=(indata.max()-indata.min())/(2**bitlength-1)
    quan=torch.round((indata-indata.min())/factor)
    dequan=quan*factor + indata.min()
    return dequan

    
def rang(x):
    dis=x.max()-x.min()
    return dis


