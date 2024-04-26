# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 13:49:50 2024

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


TSTAMP = time.strftime('%Y-%m-%d-%H-%M-%S')

RANDOM = 1
FRAME = 5

# mode_list = np.array([1,2,3,4,5],dtype=int)
mode_list = np.array([2],dtype=int)
MODE = mode_list.shape[0]

strength_list = np.array([10],dtype=float)#*(np.sqrt(2)**(5./6.))
# strength_list = np.array([0,0.1,1])#np.logspace(-1,1,10)
STRENGTH = strength_list.shape[0]

SR = np.zeros((MODE,STRENGTH,RANDOM,FRAME), dtype=float)
WFE = np.zeros((MODE,STRENGTH,RANDOM,FRAME), dtype=float)

# svd_list = np.array([[0.05,0.03,0.005],
#                      [0.05,0.03,0.005],
#                      [0.05,0.03,0.005]])
# svd_list = np.array([[0.05],
#                       [0.05],
#                       [0.05],
#                       [0.05],
#                       [0.05]])
svd_list = np.array([[0.05]])

original_name = 'sh_8x8_DM_through_turb'
original_sim = soapy.Sim(original_name + '.yaml')
original_config = original_sim.config.configDict

original_config['nIters'] = FRAME

if not os.path.exists(original_name + TSTAMP):
    os.makedirs(original_name + TSTAMP)
os.chdir(original_name + TSTAMP)

for i in np.arange(MODE):
    for j in np.arange(STRENGTH):
        new_config = dict(original_config)
        new_config['simName'] = original_config['simName'] + '{:d}-{:d}'.format(i,j)
        ndm = new_config['nDM']
        for idm in np.arange(ndm):
            if new_config['DM'][idm]['type'] == 'Aberration':
                if new_config['DM'][idm]['subtype'] == 'Zernike':
                    new_config['DM'][idm]['nollMode'] = int(mode_list[i])
                    new_config['DM'][idm]['aberrationStrength'] = float(strength_list[j])
                aberrationHeight = new_config['DM'][idm]['altitude']
        new_config['Reconstructor']['svdConditioning'] = float(svd_list[i,j])
        new_name = original_name + '{:d}-{:d}'.format(i,j)
        with open(new_name + '.yaml','w') as file:
            yaml.dump(new_config, file)
        
        SIM = soapy.Sim(new_name + '.yaml')
        SIM.aoinit()
        SIM.makeIMat(forceNew=True)
        
        TITLE = 'noll={:d},scale={:.3f}'.format(mode_list[i],strength_list[j])
        
        # plt.imshow(SIM.recon.interaction_matrix)
        # plt.colorbar()
        # plt.title('IM : {:.0f}'.format(aberrationHeight) + TITLE)
        # plt.show()
        
        # plt.imshow(SIM.recon.control_matrix)
        # plt.colorbar()
        # plt.title('CC : {:.0f}'.format(aberrationHeight) + TITLE)
        # plt.show()
    
        _,svd,_ = np.linalg.svd(SIM.recon.interaction_matrix)
        svd /= svd.max()
        SVD = svd
        
        # plt.plot(SVD,label='svd : {:d}'.format(aberrationHeight))
        # plt.hlines(SIM.recon.config.svdConditioning,0,len(svd))
        # plt.yscale('log')
        # plt.show()
        
        MASTER_IM = np.zeros((100,
                              SIM.recon.interaction_matrix.shape[0],
                              SIM.recon.interaction_matrix.shape[1]))
        MASTER_CC = np.zeros((100,
                              SIM.recon.control_matrix.shape[0],
                              SIM.recon.control_matrix.shape[1]))
        MASTER_SVD = np.zeros((100,
                               svd.shape[0]))
        
        for trial in np.arange(100):
        
            SIM = soapy.Sim(new_name + '.yaml')
            SIM.aoinit()
            SIM.makeIMat(forceNew=True)
            
            TITLE = 'noll={:d},scale={:.3f}'.format(mode_list[i],strength_list[j])
            
            plt.imshow(SIM.recon.interaction_matrix)
            plt.colorbar()
            plt.title('IM : {:.0f}'.format(aberrationHeight) + TITLE)
            plt.show()
            
            plt.imshow(SIM.recon.control_matrix)
            plt.colorbar()
            plt.title('CC : {:.0f}'.format(aberrationHeight) + TITLE)
            plt.show()
        
            _,svd,_ = np.linalg.svd(SIM.recon.interaction_matrix)
            svd /= svd.max()
            SVD = svd
            
            plt.plot(SVD,label='svd : {:d}'.format(aberrationHeight))
            plt.hlines(SIM.recon.config.svdConditioning,0,len(svd))
            plt.yscale('log')
            plt.show()
            
            MASTER_IM[trial] = SIM.recon.interaction_matrix
            MASTER_CC[trial] = SIM.recon.control_matrix
            MASTER_SVD[trial] = svd
            

        # for random in np.arange(RANDOM):
        #     SIM.aoloop()
        #     SR[i,j,random,:] = SIM.instStrehl[0]
        #     WFE[i,j,random,:] = SIM.WFE[0]
            
        #     SIM.reset_loop()
        #     for row in np.arange(SIM.atmos.scrns.shape[0]):
        #         SIM.atmos.infinite_phase_screens[row].make_initial_screen()
            
        # plt.errorbar(np.arange(SR.shape[-1]),SR[i,j].mean(0),yerr=SR[i,j].std(0),label='{:d}m'.format(aberrationHeight),marker='o',capsize=5)
        # plt.legend()
        # plt.xlabel('constant frames')
        # plt.ylabel('Strehl Ratio')
        # plt.show()
        
        # plt.errorbar(np.arange(WFE[i,j].shape[-1]), WFE[i,j].mean(0),yerr=WFE[i,j].std(0),capsize=5,label='{:}m'.format(aberrationHeight),marker='o')
        # # plt.hlines((dmfitting*(dsub/r0)**(5./3.))**0.5*500/2/np.pi,xmin=0,xmax=SR.shape[-1]-1,label='fitting error, r0={:.0f}cm'.format(r0*100))
        
        # plt.yscale('log')
        # plt.legend()
        # plt.ylabel('WFE (nm)')
        # plt.ylim(10,500)
        # # plt.ylim(10,500/np.sqrt(12))
        # plt.xlabel('time frame')
        # plt.show()


# # plt.plot(np.arange(5), sim_PHYDM0.instStrehl[0],label='PHYDM0')
# # plt.plot(np.arange(5), sim_PHYDM1.instStrehl[0],label='PHYDM1')
# # plt.plot(np.arange(5), sim_PHYDM2.instStrehl[0],label='PHYDM2')
# # plt.legend()
# # plt.xlabel('frame')
# # plt.ylabel('inst strehl')
# # plt.show()

# # # dif = np.abs(-np.log(sim_PHYDM0.instStrehl[0]) - -np.log(sim_PHYDM1.instStrehl[0]))**0.5
# # # print(dif)

# # dmfitting = -np.log(SR[0,:,-1].mean())/(dsub/r0)**(5./3.)
# # plt.title('DM fitting K = {}, r0={:.0f}cm, dsubap={:.0f}cm'.format(dmfitting,r0*100,dsub*100))
# # plt.hlines(np.exp(-0.134*(dsub/r0)**(5./3.)),xmin=0,xmax=SR.shape[-1]-1,label='Noll : r0 = {:.0f}cm'.format(r0*100),ls=':',color='r')
# # plt.hlines(np.exp(-dmfitting*(dsub/r0)**(5./3.)),xmin=0,xmax=SR.shape[-1]-1,label='DM Fitting : r0 = {:.0f}cm'.format(r0*100),ls=':',color='b')
# for i in np.arange(MODE):
#     for j in np.arange(STRENGTH):
#         plt.errorbar(np.arange(SR.shape[-1]),SR[i,j].mean(0),yerr=SR[i,j].std(0),label='{:d}m'.format(aberrationHeight),marker='o',capsize=5)
#     plt.legend()
#     plt.xlabel('constant frames')
#     plt.ylabel('Strehl Ratio')
#     plt.show()

# # plt.title('WFE of constant turbulence: r0={:.0f}cm, dsubap={:.0f}cm'.format(r0*100,dsub*100))
# for i in np.arange(MODE):
#     for j in np.arange(STRENGTH):
#         plt.errorbar(np.arange(WFE[i,j].shape[-1]), WFE[i,j].mean(0),yerr=WFE[i,j].std(0),capsize=5,label='{:}m'.format(aberrationHeight),marker='o')
#     # plt.hlines((dmfitting*(dsub/r0)**(5./3.))**0.5*500/2/np.pi,xmin=0,xmax=SR.shape[-1]-1,label='fitting error, r0={:.0f}cm'.format(r0*100))
    
#     plt.yscale('log')
#     plt.legend()
#     plt.ylabel('WFE (nm)')
#     plt.ylim(10,500/np.sqrt(12))
#     plt.xlabel('time frame')
#     plt.show()

# # plt.title('test if measured WFE is correct')
# # plt.plot(np.linspace(SR.min(),SR.max()),np.linspace(SR.min(),SR.max()),label='correct assumption')
# # plt.plot(SR.flatten(),
# #              np.exp(-(WFE.flatten()/500*2*np.pi)**2),
# #              label='{:}km'.format(heights_list[i]),ls='',marker='.',alpha=0.5)
# # plt.legend()
# # plt.ylabel('converted measured WFE')
# # plt.xlabel('measured strehl ratio')
# # plt.axis('square')
# # plt.show()

