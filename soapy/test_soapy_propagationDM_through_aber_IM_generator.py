# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:59:55 2024

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

mode_list = np.arange(6 - 1) + 2
MODE = mode_list.shape[0]
aber_r0 = 0.1
aber_L0 = 4
aberrationHeight = 5000
TRIAL = 5

original_name = 'sh_8x8_DM_through_aber_IM_generator'
original_sim = soapy.Sim(original_name + '.yaml')
original_config = original_sim.config.configDict


data = {'home' : home,
        'saveto' : saveto,
        'TSTAMP' : TSTAMP,
        'mode_list' : mode_list,
        'MODE' : MODE,
        'aber_r0' : aber_r0,
        'aber_L0' : aber_L0,
        'aberrationHeight' : aberrationHeight,
        'TRIAL' : TRIAL,
        'original_name' : original_name}

output = open(original_name + TSTAMP + '.pkl', 'wb')
pickle.dump(data,output)
output.close()

os.chdir(saveto)
output = open(original_name + TSTAMP + '.pkl', 'wb')
pickle.dump(data,output)
output.close()

os.chdir(home)

if not os.path.exists(original_name + TSTAMP):
    os.makedirs(original_name + TSTAMP)
os.chdir(original_name + TSTAMP)





# new_config = dict(original_config)
# new_config['simName'] = original_config['simName'] + 'perfect_IM' + TSTAMP
# ndm = new_config['nDM']
# for idm in np.arange(ndm):
#     if new_config['DM'][idm]['type'] == 'Aberration':
#         new_config['DM'][idm]['calibrate'] = False
#         new_config['DM'][idm]['nollMode'] = int(1)
#         new_config['DM'][idm]['r0'] = float(aber_r0)
#         new_config['DM'][idm]['L0'] = float(aber_L0)
# new_name = original_name + 'perfect_IM'
# with open(new_name + '.yaml','w') as file:
#     yaml.dump(new_config, file)

# SIM = soapy.Sim(new_name + '.yaml')
# SIM.aoinit()
# SIM.makeIMat(forceNew=True)

# TITLE = 'perfect_IM'

# IM_perfect = SIM.recon.interaction_matrix
# # MAX = IM_perfect.max()
# # MIN = IM_perfect.min()

# os.chdir(saveto)

# # plt.imshow(IM_perfect,vmin=MIN,vmax=MAX)
# # plt.colorbar()
# # plt.title('perfect_IM')
# # plt.savefig(new_name + TSTAMP + '.png', bbox_inches='tight')
# # plt.show()

# name = new_name + TSTAMP
# hdu = fits.PrimaryHDU(IM_perfect)
# hdul = fits.HDUList([hdu])

# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()

# os.chdir(home)


for i in np.arange(MODE):
    
    new_config = dict(original_config)
    new_config['simName'] = original_config['simName'] + 'zernike_bg_IM' + '{:d}'.format(i) + TSTAMP
    ndm = new_config['nDM']
    new_idm = 0
    new_dm_config = {}
    for idm in np.arange(ndm):
        if new_config['DM'][idm]['type'] == 'Aberration':
            new_config['DM'][idm]['calibrate'] = True
            new_config['DM'][idm]['nollMode'] = int(1)
            new_config['DM'][idm]['r0'] = float(aber_r0)
            new_config['DM'][idm]['L0'] = float(aber_L0)
            new_config['DM'][idm]['nollMode'] = int(mode_list[i])
            new_dm_config[new_idm] = new_config['DM'][idm]
            new_idm += 1
    new_config['nDM'] = new_idm
    new_config['DM'] = new_dm_config
    new_name = original_name + 'zernike_bg_IM'  + '{:d}'.format(i) + TSTAMP
    with open(new_name + '.yaml','w') as file:
        yaml.dump(new_config, file)
    
    SIM = soapy.Sim(new_name + '.yaml')
    SIM.aoinit()
    
    zernike_bg_IM = np.zeros((4,SIM.recon.sim_config.totalWfsData),dtype=float)
    for iscale in np.arange(4):
        scale = iscale
        # print(scale)
        phase = np.zeros((SIM.recon.n_dms, SIM.recon.scrn_size, SIM.recon.scrn_size))
        for idm in np.arange(SIM.config.sim.nDM):
            SIM.dms[idm].aberrationStrength = soapy.DM.get_noll_variance(
                SIM.config.dms[idm].diameter,
                SIM.config.dms[idm].r0,
                SIM.config.dms[idm].L0,mode_list[i])*scale
            phase[idm] = SIM.dms[idm].makeDMFrame(0)
            n_wfs_measurments = 0
            for wfs_n, wfs in SIM.recon.wfss.items():
                zernike_bg_IM[
                    iscale, n_wfs_measurments: n_wfs_measurments+wfs.n_measurements
                    ] = SIM.wfss[0].frame(
                        None,np.array([SIM.dms[0].makeDMFrame(0)]),iMatFrame=True)
                n_wfs_measurments += wfs.n_measurements
            del SIM.dms[idm].aberrationStrength, SIM.dms[idm].aberration
    
    name = new_name + TSTAMP
    TITLE = 'zernike_bg_IM' + 'noll_mode={:d},height={:.0f}km,r0={:.0f}cm,L0={:.0f}m'.format(
            mode_list[i],aberrationHeight/1000,aber_r0*100,aber_L0)
    
    for row in np.arange(zernike_bg_IM.shape[0]):
        plt.plot(zernike_bg_IM[row],label='{:d}$\sigma$'.format(row))
    plt.title(TITLE)
    plt.legend()
    plt.savefig(name + 'lage_scale' + '.png', bbox_inches='tight')
    plt.show()
    
    os.chdir(saveto)
    
    
    hdu = fits.PrimaryHDU(zernike_bg_IM)
    hdul = fits.HDUList([hdu])
    
    hdul.writeto(name + 'lage_scale' + '.fits', overwrite=True)
    hdul.close()
    
    os.chdir(home)
    
    zernike_bg_IM = np.zeros((11,SIM.recon.sim_config.totalWfsData),dtype=float)
    for iscale in np.arange(11):
        scale = iscale/10.
        # print(scale)
        phase = np.zeros((SIM.recon.n_dms, SIM.recon.scrn_size, SIM.recon.scrn_size))
        for idm in np.arange(SIM.config.sim.nDM):
            SIM.dms[idm].aberrationStrength = soapy.DM.get_noll_variance(
                SIM.config.dms[idm].diameter,
                SIM.config.dms[idm].r0,
                SIM.config.dms[idm].L0,mode_list[i])*scale
            phase[idm] = SIM.dms[idm].makeDMFrame(0)
            n_wfs_measurments = 0
            for wfs_n, wfs in SIM.recon.wfss.items():
                zernike_bg_IM[
                    iscale, n_wfs_measurments: n_wfs_measurments+wfs.n_measurements
                    ] = SIM.wfss[0].frame(
                        None,np.array([SIM.dms[0].makeDMFrame(0)]),iMatFrame=True)
                n_wfs_measurments += wfs.n_measurements
            del SIM.dms[idm].aberrationStrength, SIM.dms[idm].aberration
    
    TITLE = 'zernike_bg_IM' + 'noll_mode={:d},height={:.0f}km,r0={:.0f}cm,L0={:.0f}m'.format(
            mode_list[i],aberrationHeight/1000,aber_r0*100,aber_L0)
    
    for row in np.arange(zernike_bg_IM.shape[0]):
        plt.plot(zernike_bg_IM[row],label='{:.1f}$\sigma$'.format(row/10))
    plt.title(TITLE)
    plt.legend()
    plt.savefig(name + 'small_scale' + '.png', bbox_inches='tight')
    plt.show()
    
    os.chdir(saveto)
    
    name = new_name + TSTAMP
    hdu = fits.PrimaryHDU(zernike_bg_IM)
    hdul = fits.HDUList([hdu])
    
    hdul.writeto(name + 'small_scale' + '.fits', overwrite=True)
    hdul.close()
    
    os.chdir(home)





# MASTER_IM = np.zeros((MODE,TRIAL,
#                       SIM.recon.interaction_matrix.shape[0],
#                       SIM.recon.interaction_matrix.shape[1]))

# for i in np.arange(MODE):
#     new_config = dict(original_config)
#     new_config['simName'] = original_config['simName'] + '{:d}'.format(i) + TSTAMP
#     ndm = new_config['nDM']
#     for idm in np.arange(ndm):
#         if new_config['DM'][idm]['type'] == 'Aberration':
#             if new_config['DM'][idm]['subtype'] == 'OneZernike':
#                 new_config['DM'][idm]['nollMode'] = int(mode_list[i])
#                 new_config['DM'][idm]['r0'] = float(aber_r0)
#                 new_config['DM'][idm]['L0'] = float(aber_L0)
#                 new_config['DM'][idm]['calibrate'] = True
#             aberrationHeight = new_config['DM'][idm]['altitude']
#     new_name = original_name + '{:d}'.format(i)
#     with open(new_name + '.yaml','w') as file:
#         yaml.dump(new_config, file)
    
#     SIM = soapy.Sim(new_name + '.yaml')
#     SIM.aoinit()
#     SIM.makeIMat(forceNew=True)
    
#     TITLE = 'noll_mode={:d},height={:.0f}km,r0={:.0f}cm,L0={:.0f}m'.format(
#         mode_list[i],aberrationHeight/1000,aber_r0*100,aber_L0)
    
    
    
#     for trial in tqdm(np.arange(TRIAL)):
    
#         SIM = soapy.Sim(new_name + '.yaml')
#         SIM.aoinit()
#         SIM.makeIMat(forceNew=True)
        
#         MASTER_IM[i,trial] = SIM.recon.interaction_matrix
    
#     # IM_mean = MASTER_IM[i].mean(0)
#     # IM_std = ((MASTER_IM - MASTER_IM[i].mean(0))**2).sum(0)**0.5
    
#     # fig, ax = plt.subplots(1,2,sharex=True,sharey=True)
#     # ax[0].imshow(MASTER_IM[i].mean(0),vmin=MIN,vmax=MAX)
#     # ax[0].set_title('mean IM' + ' nollMode={:d}'.format(int(mode_list[i])))
#     # im = ax[1].imshow(IM_std,vmin=MIN,vmax=MAX)
#     # ax[1].set_title('error IM' + ' nollMode={:d}'.format(int(mode_list[i])))
    
#     # divider = make_axes_locatable(ax[1])
#     # cax = divider.append_axes("right", size="5%", pad=0.05)   
#     # plt.colorbar(im, cax=cax)
    
#     # os.chdir(saveto)
    
#     # plt.savefig(new_name + TSTAMP + '.png', bbox_inches='tight')
#     # plt.show()

#     # os.chdir(home)

# os.chdir(saveto)

# new_name = original_name + 'all_IM'

# name = new_name + TSTAMP
# hdu = fits.PrimaryHDU(MASTER_IM)
# hdul = fits.HDUList([hdu])

# hdul.writeto(name + '.fits', overwrite=True)
# hdul.close()

# MIN = IM_perfect.min()
# MAX = IM_perfect.max()

# # MIN = np.min([IM_perfect.min(),MASTER_IM.min()])
# # MAX = np.max([IM_perfect.max(),MASTER_IM.max()])

# plt.imshow(IM_perfect,vmin=MIN,vmax=MAX)
# plt.colorbar()
# plt.title('perfect_IM')
# plt.savefig(new_name + TSTAMP + 'perfect_IM' + '.png', bbox_inches='tight')
# plt.show()

# for i in np.arange(MODE):
    
#     IM_std = ((MASTER_IM[i] - IM_perfect)**2).mean(0)**0.5
    
#     # fig, ax = plt.subplots(1,2,sharex=True,sharey=True)
#     # ax[0].imshow(MASTER_IM[i].mean(0),vmin=MIN,vmax=MAX)
#     # ax[0].set_title('mean IM' + ' nollMode={:d}'.format(int(mode_list[i])))
#     # im = ax[1].imshow(IM_std,vmin=MIN,vmax=MAX)
#     # ax[1].set_title('error IM' + ' nollMode={:d}'.format(int(mode_list[i])))
    
#     # divider = make_axes_locatable(ax[1])
#     # cax = divider.append_axes("right", size="5%", pad=0.05)   
#     # plt.colorbar(im, cax=cax)
    
#     plt.imshow(IM_std,vmin=MIN,vmax=MAX)
#     plt.colorbar()
#     plt.title('error IM' + ' nollMode={:d}'.format(int(mode_list[i])))
    
#     plt.savefig(new_name + TSTAMP + 'error_IM' + 'nollMode{:d}'.format(int(mode_list[i])) + '.png', bbox_inches='tight')
#     plt.show()
    


# os.chdir(home)


