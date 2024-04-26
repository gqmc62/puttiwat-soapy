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

from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle

home = r'C:\Users\gqmc62\AppData\Local\anaconda3\Lib\site-packages\soapy\conf'
saveto = r'C:\Users\gqmc62\OneDrive - Durham University\Desktop\GLDM-IM-through-turb'

os.chdir(home)

TSTAMP = time.strftime('%Y-%m-%d-%H-%M-%S')

RANDOM = 1
FRAME = 5

mode_list = np.arange(6 - 1) + 2#np.array([2,3, 4,5,6, 7,8,9,10, 11,12,13,14,15],dtype=int)
# mode_list = np.array([6],dtype=int)
MODE = mode_list.shape[0]

strength_list = np.array([0.1],dtype=float)#*(np.sqrt(2)**(5./6.))
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
# svd_list = np.array([[0.05]])
svd_list = np.ones((mode_list.shape[0],strength_list.shape[0]),dtype=float) * 0.01

original_name = 'sh_8x8_DM_through_aber'
original_sim = soapy.Sim(original_name + '.yaml')
original_config = original_sim.config.configDict

original_config['nIters'] = FRAME

if not os.path.exists(original_name + TSTAMP):
    os.makedirs(original_name + TSTAMP)
os.chdir(original_name + TSTAMP)

new_config = dict(original_config)
new_config['simName'] = original_config['simName'] + 'perfect_IM'
ndm = new_config['nDM']
for idm in np.arange(ndm):
    if new_config['DM'][idm]['type'] == 'Aberration':
        new_config['DM'][idm]['calibrate'] = False
new_name = original_name + 'perfect_IM'
with open(new_name + '.yaml','w') as file:
    yaml.dump(new_config, file)

SIM = soapy.Sim(new_name + '.yaml')
SIM.aoinit()
SIM.makeIMat(forceNew=True)

TITLE = 'perfect_IM'

IM_perfect = SIM.recon.interaction_matrix
MAX = IM_perfect.max()
MIN = IM_perfect.min()

os.chdir(saveto)

plt.imshow(IM_perfect,vmin=MIN,vmax=MAX)
plt.colorbar()
plt.title('perfect_IM')
plt.savefig(new_name + TSTAMP + '.png', bbox_inches='tight')
plt.show()

name = new_name + 'IM' + TSTAMP
hdu = fits.PrimaryHDU(IM_perfect)
hdul = fits.HDUList([hdu])

hdul.writeto(name + '.fits', overwrite=True)
hdul.close()

os.chdir(home)

for i in np.arange(MODE):
    for j in np.arange(STRENGTH):
        new_config = dict(original_config)
        new_config['simName'] = original_config['simName'] + '{:d}-{:d}'.format(i,j)
        ndm = new_config['nDM']
        for idm in np.arange(ndm):
            if new_config['DM'][idm]['type'] == 'Aberration':
                if new_config['DM'][idm]['subtype'] == 'OneZernike':
                    new_config['DM'][idm]['nollMode'] = int(mode_list[i])
                    new_config['DM'][idm]['r0'] = float(strength_list[j])
                    new_config['DM'][idm]['L0'] = float(4)
                    new_config['DM'][idm]['calibrate'] = True
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
    
        # _,svd,_ = np.linalg.svd(SIM.recon.interaction_matrix)
        # svd /= svd.max()
        # SVD = svd
        
        # plt.plot(SVD,label='svd : {:d}'.format(aberrationHeight))
        # plt.hlines(SIM.recon.config.svdConditioning,0,len(svd))
        # plt.yscale('log')
        # plt.show()
        
        TRIAL = 1000
        
        MASTER_IM = np.zeros((TRIAL,
                              SIM.recon.interaction_matrix.shape[0],
                              SIM.recon.interaction_matrix.shape[1]))
        # MASTER_CC = np.zeros((TRIAL,
        #                       SIM.recon.control_matrix.shape[0],
        #                       SIM.recon.control_matrix.shape[1]))
        # MASTER_SVD = np.zeros((TRIAL,
        #                         svd.shape[0]))
        
        for trial in tqdm(np.arange(TRIAL)):
        
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
        
            # _,svd,_ = np.linalg.svd(SIM.recon.interaction_matrix)
            # svd /= svd.max()
            # SVD = svd
            
            # plt.plot(SVD,label='svd : {:d}'.format(aberrationHeight))
            # plt.hlines(SIM.recon.config.svdConditioning,0,len(svd))
            # plt.yscale('log')
            # plt.show()
            
            MASTER_IM[trial] = SIM.recon.interaction_matrix
            # MASTER_CC[trial] = SIM.recon.control_matrix
            # MASTER_SVD[trial] = svd
        
        IM_mean = MASTER_IM.mean(0)
        IM_std = ((MASTER_IM - MASTER_IM.mean(0))**2).sum(0)**0.5
        # MAX = IM_mean.max()
        # MIN = IM_mean.min()
        
        fig, ax = plt.subplots(1,2,sharex=True,sharey=True)
        ax[0].imshow(MASTER_IM.mean(0),vmin=MIN,vmax=MAX)
        ax[0].set_title('mean IM' + ' nollMode={:d}'.format(int(mode_list[i])))
        im = ax[1].imshow(IM_std,vmin=MIN,vmax=MAX)
        ax[1].set_title('error IM' + ' nollMode={:d}'.format(int(mode_list[i])))
        
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)   
        plt.colorbar(im, cax=cax)
        
        os.chdir(saveto)
        
        plt.savefig(new_name + TSTAMP + '.png', bbox_inches='tight')
        plt.show()
        
        # data = {'IM_mean' : IM_mean,
        #         'IM_std' : IM_std}
        # output = open(new_name + TSTAMP + '.pkl', 'wb')
        # pickle.dump(data,output)
        # output.close()
        
        name = new_name + 'IM' + TSTAMP
        hdu = fits.PrimaryHDU(MASTER_IM)
        hdul = fits.HDUList([hdu])
        
        hdul.writeto(name + '.fits', overwrite=True)
        hdul.close()
        
        # name = new_name + 'CC' + TSTAMP
        # hdu = fits.PrimaryHDU(MASTER_CC)
        # hdul = fits.HDUList([hdu])
        
        # hdul.writeto(name + '.fits', overwrite=True)
        # hdul.close()
        
        os.chdir(home)

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

