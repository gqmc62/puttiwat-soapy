# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:50:27 2023

@author: gqmc62
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 12:44:04 2023

@author: gqmc62
"""

from matplotlib import pyplot as plt
from soapy import soapy
import os
import numpy as np
from tqdm import tqdm
os.chdir(r'C:\Users\gqmc62\AppData\Local\anaconda3\Lib\site-packages\soapy\conf')

RANDOM = 2
HT = 5 + 1
FRAME = 5
SR = np.zeros((HT,RANDOM,FRAME), dtype=float)

SIMS = []
SVDS = []
for i in np.arange(HT):
    SIMS.append(soapy.Sim('sh_8x8_PHYDM{:d}.yaml'.format(i)))

for i in np.arange(HT):
    SIMS[i].aoinit()
    SIMS[i].makeIMat(forceNew=True)

    plt.imshow(SIMS[i].recon.interaction_matrix)
    plt.colorbar()
    plt.title('IM : {:d}'.format(i))
    plt.show()
    plt.imshow(SIMS[i].recon.control_matrix)
    plt.colorbar()
    plt.title('CC : {:d}'.format(i))
    plt.show()
    
    _,svd,_ = np.linalg.svd(SIMS[i].recon.interaction_matrix)
    svd /= svd.max()
    SVDS.append(svd)

for i in np.arange(HT):
    plt.plot(SVDS[i],label='svd : {:d}'.format(i))
plt.hlines(SIMS[0].recon.config.svdConditioning,0,len(svd),label='threshold')
plt.yscale('log')
plt.title('normalized svd')
plt.legend()
plt.show()


for i in np.arnge(HT):
    for random in np.arange(RANDOM):
        SIMS[i].aoloop()
        SR[i,random,:] = SIMS[i].instStrehl[0]
        
        SIMS[i].reset_loop()
        for j in np.arange(SIMS[i].atmos.scrns.shape[0]):
            SIMS[i].atmos.infinite_phase_screens[j].make_initial_screen()


# plt.plot(np.arange(5), sim_PHYDM0.instStrehl[0],label='PHYDM0')
# plt.plot(np.arange(5), sim_PHYDM1.instStrehl[0],label='PHYDM1')
# plt.plot(np.arange(5), sim_PHYDM2.instStrehl[0],label='PHYDM2')
# plt.legend()
# plt.xlabel('frame')
# plt.ylabel('inst strehl')
# plt.show()

# # dif = np.abs(-np.log(sim_PHYDM0.instStrehl[0]) - -np.log(sim_PHYDM1.instStrehl[0]))**0.5
# # print(dif)

plt.title('DM fitting K = {}'.format(-np.log(SR[0,:,-1].mean())))
plt.hlines(np.exp(-0.134),xmin=0,xmax=SR.shape[-1]-1,label='Noll')
for i in np.arange(HT):
    plt.errorbar(np.arange(SR.shape[-1]),SR[i].mean(0),yerr=SR[i].std(0),label='{:d}km'.format(i),marker='o',capsize=5)
plt.legend()
plt.xlabel('constant frames')
plt.ylabel('Strehl Ratio')
plt.show()



