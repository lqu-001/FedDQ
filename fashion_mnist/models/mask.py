#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import scipy.sparse as spy
import numpy as np
    
def quandequan(indata, bitlength):
    import numpy as np
    import torch
    import torch.nn.functional as F
    indata=torch.tensor(indata)
    factor=(indata.max()-indata.min())/(2**bitlength-1)
    quan=F.hardtanh(torch.round(indata/factor), min_val=-2**(bitlength-1), max_val=2**(bitlength-1)-1)
    dequan=quan*factor 
    return dequan
    
def rang(x):
    dis=np.max(x)-np.min(x)
    return dis


