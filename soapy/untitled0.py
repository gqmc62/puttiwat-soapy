# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 18:14:49 2023

@author: gqmc62
"""

from matplotlib import pyplot as plt
from soapy import soapy
import os
import numpy as np
os.chdir(r'C:\Users\gqmc62\AppData\Local\anaconda3\Lib\site-packages\soapy\conf')
sim_PHY = soapy.Sim('sh_8x8_PHY.yaml')
sim_PHY.aoinit()
sim_PHY.makeIMat(forceNew=True)
sim_PHY.aoloop()
# for time in np.arange(10):
#     sim_PHY.loopFrame()
#     plt.imshow(sim_PHY.wfss[0].wfsDetectorPlane)
#     plt.show()