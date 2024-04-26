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
import yaml
import time
os.chdir(r'C:\Users\gqmc62\AppData\Local\anaconda3\Lib\site-packages\soapy\conf')
from astropy.io import fits


def get_rcond_adaptive_threshold_rank(A,plot=False,return_place=False):
  _,s,_ = np.linalg.svd(A)
  s /= s.max()
  energy = s**2
  energy /= energy.sum()
  accumulated_energy = np.zeros_like(energy)
  threshold = 1/2./np.linalg.matrix_rank(A)

  for i in np.arange(energy.shape[0]):
    accumulated_energy[i] = energy[:i].sum()
  residual = 1 - accumulated_energy
  place = np.where(residual<=threshold)[0][0]
  
  if plot==True:
    # plt.plot(residual,label='residual')
    # plt.hlines(residual[place],0,s.shape[0])
    # plt.vlines(place,0,1)
    # plt.yscale('log')
    # plt.show()

    plt.plot(s)
    plt.hlines(s[place],0,s.shape[0])
    plt.vlines(place,0,1)
    plt.yscale('log')
    plt.show()
  rcond = s[place]
  if return_place == True:
    return rcond, place
  else:
    return rcond

def pinv_adaptive_threshold_rank(A,return_rcond=False):
    if return_rcond == False:
        rcond = get_rcond_adaptive_threshold_rank(A)
        pinvA = np.linalg.pinv(A,rcond=rcond)
        return pinvA
    else:
        rcond, place = get_rcond_adaptive_threshold_rank(A,return_place=True)
        return pinvA, rcond, place



# FRAME = 50
# RANDOM = 50

heights_list = [10000]
#[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]#[2000]#[0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
#[1000]#[0,0,0,0,0]
#[0,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]

HT = len(heights_list)

r0_list = [0.05]#[0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]
#[0.08,0.13,0.13,0.16,0.16,0.16,0.16,0.16,0.32,0.32,0.32]#[0.10,0.08,0.05,0.04]#[0.10,0.05,0.02]
#[0.13]#[0.08,0.08,0.08,0.08,0.08]
#[0.08,0.13,0.13,0.16,0.16,0.16,0.16,0.16,0.32,0.32,0.32]#[0.10,0.08,0.05,0.04]#[0.10,0.05,0.02]#[::-1]#[0.05]#
nxsubap = 8
sci_wavelength = 500e-9#1500e-9
TSTAMP = time.strftime('%Y-%m-%d-%H-%M-%S')
    
gain = 0.5

CLOSE = True

FRAME = 5
RANDOM = 1
# if CLOSE == False:
#     FRAME = 10#10
#     RANDOM = 10#50
# else:
#     FRAME = 50#50
#     RANDOM = 10#10
    
os.chdir(r'C:\Users\gqmc62\AppData\Local\anaconda3\Lib\site-packages\soapy\conf')
SR = np.zeros((HT,RANDOM,FRAME), dtype=float)
WFE = np.zeros((HT,RANDOM,FRAME), dtype=float)




original_name = 'sh_8x8_PHYDM'

original_sim = soapy.Sim(original_name + '.yaml')
original_config = original_sim.config.configDict

original_config['nIters'] = FRAME

for isci in np.arange(original_config['nSci']):
    original_config['Science'][isci]['wavelength'] = sci_wavelength
for iwfs in np.arange(original_config['nGS']):
    original_config['WFS'][iwfs]['nxSubaps'] = nxsubap
dsub = original_sim.config.tel.telDiam/nxsubap
# if 3*r0 < dsub:
#     break
for idm in np.arange(original_config['nDM']):
    original_config['DM'][idm]['closed'] = CLOSE
    if CLOSE:
        original_config['DM'][idm]['gain'] = gain
    else:
        original_config['DM'][idm]['gain'] = 1
        original_config['nIters'] = 10
    
original_config['Reconstructor']['gain'] = gain
# original_config['Reconstructor']['svdConditioning'] = svd_level




wfsSubapFov = int(np.ceil(original_sim.config.wfss[0].pxlsPerSubap
                          * original_sim.config.wfss[0].wavelength/2./dsub
                          *(3600*360/2./np.pi)))
for iwfs in np.arange(original_config['nGS']):
    original_config['WFS'][iwfs]['subapFOV'] = wfsSubapFov
    
original_config['pupilSize'] = int(original_config['WFS'][iwfs]['nxSubaps']*original_config['WFS'][iwfs]['pxlsPerSubap'])





if not os.path.exists(original_name + TSTAMP):
    os.makedirs(original_name + TSTAMP)
os.chdir(original_name + TSTAMP)

for i in np.arange(HT):
    new_config = dict(original_config)
    new_config['simName'] = original_config['simName'] + '{:d}'.format(heights_list[i])
    ndm = new_config['nDM']
    for idm in np.arange(ndm):
        if new_config['DM'][idm]['type'] != 'TT':
            new_config['DM'][idm]['altitude'] = float(heights_list[i])
    new_config['Atmosphere']['scrnHeights'] = [float(heights_list[i])]
    new_config['Atmosphere']['r0'] = float(r0_list[i])
    new_name = original_name + '{:d}'.format(i)
    with open(new_name+'.yaml','w') as file:
        yaml.dump(new_config, file)

try:
    SIMS
    for i in np.arange(HT):
        if SIMS[i].config.atmos.r0 != r0_list[i]:
            SIMS[i].config.atmos.r0 = r0_list[i]
            SIMS[i].atmos = soapy.atmosphere.atmos(SIMS[i].config)
            
        SIMS[i].reset_loop()
        SIMS[i].initSaveData()

        # Init simulation
        #Circular buffers to hold loop iteration correction data
        SIMS[i].slopes = np.zeros((SIMS[i].config.sim.totalWfsData))
        SIMS[i].closed_correction = np.zeros((
                SIMS[i].config.sim.nDM, SIMS[i].config.sim.scrnSize, SIMS[i].config.sim.scrnSize
                ))
        SIMS[i].open_correction = SIMS[i].closed_correction.copy()
        SIMS[i].dmCommands = np.zeros(SIMS[i].config.sim.totalActs)
        # SIMS[i].buffer = soapy.buffer.DelayBuffer()
        SIMS[i].iters = 0

        # Init performance tracking
        SIMS[i].Twfs = 0
        SIMS[i].Tlgs = 0
        SIMS[i].Tdm = 0
        SIMS[i].Tsci = 0
        SIMS[i].Trecon = 0
        SIMS[i].Timat = 0
        SIMS[i].Tatmos = 0
            
            
except:
    SIMS = []
    SVDS = []
    for i in np.arange(HT):
        SIMS.append(soapy.Sim(original_name + '{:d}.yaml'.format(i)))
    
    for i in np.arange(HT):
        SIMS[i].aoinit()
        
    for i in np.arange(HT):
        SIMS[i].makeIMat(forceNew=True)
        
        TITLE = '\nr0={:.0f}cm,dsub={:.0f}cm,dm@{:.0f}m'.format(
            SIMS[i].config.atmos.r0*100,
            SIMS[i].config.tel.telDiam/SIMS[i].config.wfss[0].nxSubaps*100,
            SIMS[i].config.dms[-1].altitude)
        plt.imshow(SIMS[i].recon.interaction_matrix)
        plt.colorbar()
        plt.title('IM : {:.0f}'.format(heights_list[i]) + TITLE)
        plt.show()
        plt.imshow(SIMS[i].recon.control_matrix)
        plt.colorbar()
        plt.title('CC : {:.0f}'.format(heights_list[i]) + TITLE)
        plt.show()
        
        _,svd,_ = np.linalg.svd(SIMS[i].recon.interaction_matrix)
        svd /= svd.max()
        SVDS.append(svd)
        
        plt.plot(SVDS[i],label='svd : {:d}'.format(heights_list[i]))
        plt.hlines(SIMS[i].recon.config.svdConditioning,0,len(svd))
        plt.yscale('log')
        plt.show()
    
    for i in np.arange(HT):
        plt.plot(SVDS[i],label='svd : {:d}'.format(heights_list[i]))
        # plt.hlines(SIMS[i].recon.config.svdConditioning,0,len(svd))#,
                   #label='threshold : {:d}'.format(heights_list[i]),ls=':')
    plt.yscale('log')
    TITLE = '\nr0={:.0f}cm,dsub={:.0f}cm='.format(
        SIMS[i].config.atmos.r0*100,
        SIMS[i].config.tel.telDiam/SIMS[i].config.wfss[0].nxSubaps*100)
    plt.title('normalized svd' + TITLE)
    plt.legend()
    plt.show()


for i in np.arange(HT):
    for random in np.arange(RANDOM):
        SIMS[i].aoloop()
        SR[i,random,:] = SIMS[i].instStrehl[0]
        WFE[i,random,:] = SIMS[i].WFE[0]
        
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

# dmfitting = -np.log(SR[0,:,-1].mean())/(dsub/r0)**(5./3.)
# plt.title('DM fitting K = {}, r0={:.0f}cm, dsubap={:.0f}cm'.format(dmfitting,r0*100,dsub*100))
# plt.hlines(np.exp(-0.134*(dsub/r0)**(5./3.)),xmin=0,xmax=SR.shape[-1]-1,label='Noll : r0 = {:.0f}cm'.format(r0*100),ls=':',color='r')
# plt.hlines(np.exp(-dmfitting*(dsub/r0)**(5./3.)),xmin=0,xmax=SR.shape[-1]-1,label='DM Fitting : r0 = {:.0f}cm'.format(r0*100),ls=':',color='b')
for i in np.arange(HT):
    plt.errorbar(np.arange(SR.shape[-1]),SR[i].mean(0),yerr=SR[i].std(0),label='{:d}m'.format(heights_list[i]),marker='o',capsize=5)
plt.legend()
plt.xlabel('constant frames')
plt.ylabel('Strehl Ratio')
plt.show()

# plt.title('WFE of constant turbulence: r0={:.0f}cm, dsubap={:.0f}cm'.format(r0*100,dsub*100))
for i in np.arange(HT):
    plt.errorbar(np.arange(WFE[i].shape[-1]), WFE[i].mean(0),yerr=WFE[i].std(0),capsize=5,label='{:}m'.format(heights_list[i]),marker='o')
# plt.hlines((dmfitting*(dsub/r0)**(5./3.))**0.5*500/2/np.pi,xmin=0,xmax=SR.shape[-1]-1,label='fitting error, r0={:.0f}cm'.format(r0*100))

plt.yscale('log')
plt.legend()
plt.ylabel('WFE (nm)')
plt.ylim(10,sci_wavelength/1e-9/np.sqrt(12))
plt.xlabel('time frame')
plt.show()

# plt.title('test if measured WFE is correct')
# plt.plot(np.linspace(SR.min(),SR.max()),np.linspace(SR.min(),SR.max()),label='correct assumption')
# plt.plot(SR.flatten(),
#              np.exp(-(WFE.flatten()/500*2*np.pi)**2),
#              label='{:}km'.format(heights_list[i]),ls='',marker='.',alpha=0.5)
# plt.legend()
# plt.ylabel('converted measured WFE')
# plt.xlabel('measured strehl ratio')
# plt.axis('square')
# plt.show()

