#Copyright Durham University and Andrew Reeves
#2014

# This file is part of soapy.

#     soapy is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     soapy is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with soapy.  If not, see <http://www.gnu.org/licenses/>.

import traceback
import time

import numpy
from matplotlib import pyplot as plt
from astropy.io import fits

from . import logger

from . import interp
from scipy.signal import convolve2d
from scipy.optimize import curve_fit
import os

# Use pyfits or astropy for fits file handling
try:
    from astropy.io import fits
except ImportError:
    try:
        import pyfits as fits
    except ImportError:
        raise ImportError("soapy requires either pyfits or astropy")

#xrange now just "range" in python3.
#Following code means fastest implementation used in 2 and 3
try:
    xrange
except NameError:
    xrange = range
    
def bin_this(A, w):
    # a = numpy.nanmean(A[:(A.shape[0]//w)*w,
    #       :(A.shape[1]//w)*w].reshape(
    #           (((A.shape[0]//w),w,((A.shape[1]//w)),w))
    #           ), axis=(1,3))
    a = numpy.mean(A[:(A.shape[0]//w)*w,
          :(A.shape[1]//w)*w].reshape(
              (((A.shape[0]//w),w,((A.shape[1]//w)),w))
              ), axis=(1,3))
    return a

class Reconstructor(object):
    """
    Reconstructor that will give DM commands required to correct an AO frame for a given set of WFS measurements
    """
    def __init__(self, soapy_config, dms, wfss, atmos, runWfsFunc=None):

        self.soapy_config = soapy_config

        self.dms = dms
        self.wfss = wfss
        self.sim_config = soapy_config.sim
        self.atmos = atmos
        self.config = soapy_config.recon

        self.n_dms = soapy_config.sim.nDM
        self.scrn_size = soapy_config.sim.scrnSize

        self.learnIters = self.sim_config.learnIters

        self.dmActs = []
        self.dmConds = []
        self.dmTypes = []
        for dm in xrange(self.sim_config.nDM):
            self.dmActs.append(self.dms[dm].dmConfig.nxActuators)
            self.dmConds.append(self.dms[dm].dmConfig.svdConditioning)
            self.dmTypes.append(self.dms[dm].dmConfig.type)

        self.dmConds = numpy.array(self.dmConds)
        self.dmActs = numpy.array(self.dmActs)

        n_acts = 0
        self.first_acts = []
        for i, dm in self.dms.items():
            self.first_acts.append(n_acts)
            n_acts += dm.n_acts

        n_wfs_measurements = 0
        self.first_measurements = []
        for i, wfs in self.wfss.items():
            self.first_measurements.append(n_wfs_measurements)
            n_wfs_measurements += wfs.n_measurements

        #2 functions used in case reconstructor requires more WFS data.
        #i.e. learn and apply
        self.runWfs = runWfsFunc
        if self.sim_config.learnAtmos == "random":
            self.moveScrns = atmos.randomScrns
        else:
            self.moveScrns = atmos.moveScrns
        self.wfss = wfss

        self.control_matrix = numpy.zeros(
            (self.sim_config.totalWfsData, self.sim_config.totalActs))
        self.controlShape = (
            self.sim_config.totalWfsData, self.sim_config.totalActs)

        self.Trecon = 0

        self.find_closed_actuators()

        self.actuator_values = None

    def find_closed_actuators(self):
        self.closed_actuators = numpy.zeros(self.sim_config.totalActs)
        n_act = 0
        for i_dm, dm in self.dms.items():
            if dm.dmConfig.closed:
                self.closed_actuators[n_act: n_act + dm.n_acts] = 1
            n_act += dm.n_acts

    def saveCMat(self):
        """
        Writes the current control Matrix to FITS file
        """
        filename = self.sim_config.simName+"/cMat.fits"

        fits.writeto(
                filename, self.control_matrix,
                header=self.sim_config.saveHeader, overwrite=True)

    def loadCMat(self):
        """
        Loads a control matrix from file to the reconstructor

        Looks in the standard reconstructor directory for a control matrix and loads the file.
        Also looks at the FITS header and checks that the control matrix is compatible with the current simulation.
        """

        filename = self.sim_config.simName+"/cMat.fits"

        logger.info("Load Command Matrix")

        cMatHDU = fits.open(filename)[0]
        cMatHDU.verify("fix")
        header = cMatHDU.header

        try:
            dmNo = int(header["NBDM"])
            exec("dmActs = numpy.array({})".format(
                    cMatHDU.header["DMACTS"]), globals())
            exec("dmTypes = %s" % header["DMTYPE"], globals())
            exec("dmConds = numpy.array({})".format(
                    cMatHDU.header["DMCOND"]), globals())

            if not numpy.allclose(dmConds, self.dmConds):
                raise IOError("DM conditioning Parameter changed - will make new control matrix")
            if not numpy.all(dmActs==self.dmActs) or dmTypes!=self.dmTypes or dmNo!=dmNo:
                logger.warning("loaded control matrix may not be compatibile with \
                                the current simulation. Will try anyway....")

            cMat = cMatHDU.data

        except KeyError:
            logger.warning("loaded control matrix header has not created by this ao sim. Will load anyway.....")
            #cMat = cMatFile[1]
            cMat = cMatHDU.data

        if cMat.shape != self.controlShape:
            logger.warning("designated control matrix does not match the expected shape")
            raise IOError
        else:
            self.control_matrix = cMat

    def save_interaction_matrix(self):
        """
        Writes the current control Matrix to FITS file

        Writes the current interaction matrix to a FITS file in the simulation directory. Also
        writes the "valid actuators" as accompanying FITS files, and potentially premade DM
        influence functions.
        """
        imat_filename = self.sim_config.simName+"/iMat.fits"

        fits.writeto(
                imat_filename, self.interaction_matrix,
                header=self.sim_config.saveHeader, overwrite=True)

        for i in range(self.n_dms):
            valid_acts_filename =  self.sim_config.simName+"/active_acts_dm{}.fits".format(i)
            valid_acts = self.dms[i].valid_actuators
            fits.writeto(valid_acts_filename, valid_acts, header=self.sim_config.saveHeader, overwrite=True)

            # If DM has pre made influence funcs, save them too
            try:
                dm_shapes_filename = self.sim_config.simName + "/dmShapes_dm{}.fits".format(i)
                fits.writeto(
                        dm_shapes_filename, self.dms[i].iMatShapes,
                        header=self.simConfig.saveHeader, overwrite=True)
            # If not, don't worry about it! Must be a DM with no pre-made influence funcs
            except AttributeError:
                pass
                
    def load_interaction_matrix(self):
        """
        Loads the interaction matrix from file

        AO interaction matrices can get very big, so its useful to be able to load it frmo file
        rather than make it new everytime. It is assumed that the iMat is saved as "FITS" in the
        simulation saved directory, with other accompanying FITS files that contain the indices
        of actuators which are "valid". Some DMs also use pre made influence functions, which are
        also loaded here.
        """

        filename = self.sim_config.simName+"/iMat.fits"

        imat_header = fits.getheader(filename)
        imat_data = fits.getdata(filename)

        imat_totalActs = imat_header['DMNACTU']
        imat_totalWfsData = imat_header['NSLOP']
        
        # Check iMat generated with same totalActs and totalWfsData as current sim
        # NOTE the actual shape of the loaded iMat can be different due to invalid acts
        if imat_totalActs != self.sim_config.totalActs or imat_totalWfsData != self.sim_config.totalWfsData:
            logger.warning(
                "interaction matrix not generated with same number of actuators/wfs slopes"
            )
            raise IOError("interaction matrix does not match required required size.")

        # Load valid actuators
        n_total_valid_acts = 0
        for i in range(self.n_dms):
            valid_acts_filename =  self.sim_config.simName+"/active_acts_dm{}.fits".format(i)
            valid_acts = fits.getdata(valid_acts_filename)
            self.dms[i].valid_actuators = valid_acts
            n_total_valid_acts += self.dms[i].n_valid_actuators

            # DM may also have preloaded influence functions
            try:
                dm_shapes_filename = self.sim_config.simName + "/dmShapes_dm{}.fits".format(i)
                dm_shapes = fits.getdata(dm_shapes_filename)
                self.dms[i].iMatShapes = dm_shapes

            except IOError:
                # Found no DM influence funcs
                logger.info("DM Influence functions not found. If the DM doesn't use them, this is ok. If not, set 'forceNew=True' when making IMat")

        # Final check of loaded iMat
        if imat_data.shape != (n_total_valid_acts, self.sim_config.totalWfsData):
            logger.warning(
                "interaction matrix does not match required required size."
            )
            raise IOError("interaction matrix does not match required required size.")

        self.interaction_matrix = imat_data

    def makeIMat(self, callback=None):

        self.interaction_matrix = numpy.zeros((self.sim_config.totalActs, self.sim_config.totalWfsData))

        n_acts = 0
        dm_imats = []
        total_valid_actuators = 0
        for dm_n, dm in self.dms.items():
            logger.info("Creating Interaction Matrix for DM %d " % (dm_n))
            if dm.config.type != 'Aberration':
                dm_imats.append(self.make_dm_iMat(dm, callback=callback))
    
                total_valid_actuators += dm_imats[dm_n].shape[0]

        self.interaction_matrix = numpy.zeros((total_valid_actuators, self.sim_config.totalWfsData))
        act_n = 0
        for imat in dm_imats:
            self.interaction_matrix[act_n: act_n + imat.shape[0]] = imat
            act_n += imat.shape[0]
            
        # plt.imshow(self.interaction_matrix)
        # plt.title('interaction matrix')
        # plt.show()


    def make_dm_iMat(self, dm, callback=None):
        """
        Makes an interaction matrix for a given DM with a given WFS

        Parameters:
            dm (DM): The Soapy DM for which an interaction matri is required.
            wfs (WFS): The Soapy WFS for which an interaction matrix is required
            callback (func, optional): A function to be called each iteration accepting no arguments
        """

        iMat = numpy.zeros(
            (dm.n_acts, self.sim_config.totalWfsData))

        # A vector of DM commands to use when making the iMat
        actCommands = numpy.zeros(dm.n_acts)

        phase = numpy.zeros((self.n_dms, self.scrn_size, self.scrn_size))
        
        
        ABERRATION = False
        
        # zero poke case
        
        zero_iMat = numpy.zeros((self.sim_config.totalWfsData))
        zero_wfs_efield = []
        
        # Set vector of iMat commands and phase to 0
        actCommands[:] = 0
        # Now get a DM shape for that command
        phase[:] = 0
        phase[dm.n_dm] = dm.dmFrame(actCommands)
        for DM_N, DM in self.dms.items():
            if DM.config.type == 'Aberration':
                ABERRATION = True
                if DM.config.calibrate == False:
                    phase[DM_N] = DM.dmFrame(1)
                else:
                    phase[DM_N] = DM.dmFrame(0)
        # Send the DM shape off to the relavent WFS. put result in iMat
        n_wfs_measurments = 0
        for wfs_n, wfs in self.wfss.items():
            # turn off wfs noise if set
            if self.config.imat_noise is False:
                wfs_pnoise = wfs.config.photonNoise
                wfs.config.photonNoise = False
                wfs_rnoise = wfs.config.eReadNoise
                wfs.config.eReadNoise = 0
            
            zero_iMat[n_wfs_measurments: n_wfs_measurments+wfs.n_measurements] = (
                    wfs.frame(None, phase_correction=phase, iMatFrame=True))# / dm.dmConfig.iMatValue
        
            zero_wfs_efield.append(numpy.copy(wfs.interp_efield))
        
        # plt.plot(zero_iMat)
        # plt.show()
        
        self.zero_iMat = zero_iMat
        
        # poke each dms
        
        setattr(dm,'subapShift', numpy.zeros((dm.n_acts,len(self.wfss),2)))
        dm.subapShift[:] = numpy.nan
        setattr(dm,'rytov', numpy.zeros((dm.n_acts,len(self.wfss))))
        
        for i in range(dm.n_acts):
            # Set vector of iMat commands and phase to 0
            actCommands[:] = 0

            # Except the one we want to make an iMat for!
            actCommands[i] = 1 # dm.dmConfig.iMatValue

            # Now get a DM shape for that command
            phase[:] = 0
            phase[dm.n_dm] = dm.dmFrame(actCommands)
            for DM_N, DM in self.dms.items():
                if DM.config.type == 'Aberration':
                    if DM.config.calibrate == False:
                        phase[DM_N] = DM.dmFrame(1)
                    else:
                        phase[DM_N] = DM.dmFrame(0)
            # Send the DM shape off to the relavent WFS. put result in iMat
            n_wfs_measurments = 0
            for wfs_n, wfs in self.wfss.items():
                # turn off wfs noise if set
                if self.config.imat_noise is False:
                    wfs_pnoise = wfs.config.photonNoise
                    wfs.config.photonNoise = False
                    wfs_rnoise = wfs.config.eReadNoise
                    wfs.config.eReadNoise = 0
                
                plot = False
                if (plot == True) and (i == 40):
                #     plt.imshow(phase.sum(0))
                #     plt.title('all geom dm')
                #     plt.colorbar()
                #     plt.show()
                    actCommands = numpy.zeros((81),dtype=float)
                    actCommands[20] = 1
                    actCommands[22] = 1
                    actCommands[24] = 1
                    actCommands[38] = 1
                    actCommands[40] = 1
                    actCommands[42] = 1
                    actCommands[56] = 1
                    actCommands[58] = 1
                    actCommands[60] = 1
                    phase[dm.n_dm] = dm.dmFrame(actCommands)
                    wfs.frame(None, phase_correction=phase, iMatFrame=True)
                    
                    xx = numpy.arange(numpy.array(zero_wfs_efield[wfs_n]).shape[0],dtype=float)
                    xx -= xx.max()/2.
                    xx /= wfs.nx_subap_interp
                    
                    zero_wfs_efield[wfs_n][wfs.scaledMask == 0] = numpy.nan
                    
                    
                    wfs_size = wfs.interp_efield.shape[0]
                    
                    A = phase.shape[1]
                    B = self.soapy_config.sim.pupilSize
                    P = (A - B)//2
                    Q = (A + B)//2
                    
                    no_aberration_efield = (numpy.exp(1j*interp.zoom(phase[:-1,P:Q,P:Q].sum(0),wfs_size)/500.*2*numpy.pi))
                    
                    wfs.interp_efield[wfs.scaledMask == 0] = numpy.nan
                    
                    with_aberration_efield = (wfs.interp_efield
                                            / numpy.asarray(zero_wfs_efield[wfs_n]))
                    
                    to_plot = numpy.angle(with_aberration_efield)
                    to_plot -= numpy.nanmedian(to_plot)
                    fig, ax1 = plt.subplots()
                    
                    c = ax1.pcolor(xx,xx,
                                -to_plot,vmin=0,vmax=0.012)
                    
                    ax1.hlines((-4,-3,-2,-1,0,1,2,3,4),xmin=-4,xmax=4,color='w')
                    ax1.vlines((-4,-3,-2,-1,0,1,2,3,4),ymin=-4,ymax=4,color='w')
                    ax1.plot([0,2,-2,0,0,-2,-2,2,2],[0,0,0,2,-2,-2,2,-2,2],color='k',marker='o',ls='')
                    fig.colorbar(c,ax=ax1)
                    ax1.axis('square')
                    
                    frame1 = ax1
                    for xlabel_i in frame1.axes.get_xticklabels():
                        xlabel_i.set_visible(False)
                        xlabel_i.set_fontsize(0.0)
                    for xlabel_i in frame1.axes.get_yticklabels():
                        xlabel_i.set_fontsize(0.0)
                        xlabel_i.set_visible(False)
                        
                    ax1.tick_params(axis='both', which='both', length=0)
                    
                    plt.gcf().set_size_inches(6,6)
                    
                    plt.savefig('dm_influence-' + time.strftime("%Y-%m-%d-%H-%M-%S") + '.png',
                                dpi=300,bbox_inches='tight',transparent=True)
                
                    plt.show()
                    
                    actCommands[:] = 0

                    # Except the one we want to make an iMat for!
                    actCommands[i] = 1 # dm.dmConfig.iMatValue
                    phase[dm.n_dm] = dm.dmFrame(actCommands)
                
                iMat[i, n_wfs_measurments: n_wfs_measurments+wfs.n_measurements] = -1 * (
                    wfs.frame(scrns=None, phase_correction=phase, iMatFrame=True)
                    - zero_iMat[n_wfs_measurments: n_wfs_measurments+wfs.n_measurements])# / dm.dmConfig.iMatValue
                # # plt.plot(wfs.frame(None, phase_correction=phase, iMatFrame=True),label='wfs measurement')
                # # plt.plot(zero_iMat[n_wfs_measurments: n_wfs_measurments+wfs.n_measurements],label='zero')
                # # plt.plot(iMat[i, n_wfs_measurments: n_wfs_measurments+wfs.n_measurements],label='IM')
                # # plt.legend()
                # # plt.show()
                
                # # if i == 40:
                # #     plt.imshow(phase.sum(0))
                # #     plt.title('all geom dm')
                # #     plt.colorbar()
                # #     plt.show()
                
                # xx = numpy.arange(numpy.array(zero_wfs_efield[wfs_n]).shape[0],dtype=float)
                # xx -= xx.max()/2.
                # xx /= wfs.nx_subap_interp
                
                # zero_wfs_efield[wfs_n][wfs.scaledMask == 0] = numpy.nan
                
                # # plt.pcolor(xx,xx,numpy.angle(numpy.asarray(zero_wfs_efield[wfs_n])).T)
                # # plt.axis('square')
                # # plt.title('0 phase')
                # # plt.colorbar()
                # # plt.show()
                
                # # hdu = fits.PrimaryHDU([numpy.exp(1j*phase[:-1,129:257,129:257].sum(0)/500.*2*3.14).real,
                # #                       numpy.exp(1j*phase[:-1,129:257,129:257].sum(0)/500.*2*3.14).imag])
                # # hdul = fits.HDUList([hdu])
                # # hdul.writeto('poke{:d}.fits'.format(i),overwrite=True)
                # # # hdu.close()
                
                # wfs_size = wfs.interp_efield.shape[0]
                
                # A = phase.shape[1]
                # B = self.soapy_config.sim.pupilSize
                # P = (A - B)//2
                # Q = (A + B)//2
                
                # no_aberration_efield = (numpy.exp(1j*interp.zoom(phase[:-1,P:Q,P:Q].sum(0),wfs_size)/500.*2*numpy.pi))
                # min_phase = -0.012#(-numpy.angle(no_aberration_efield)).min()
                # max_phase = 0#(-numpy.angle(no_aberration_efield)).max()
                # # no_aberration_efield /= numpy.nanmean(no_aberration_efield)
                # # no_aberration_efield /= numpy.sqrt(numpy.nanmean(numpy.abs(no_aberration_efield)**2))
                # no_aberration = numpy.angle(no_aberration_efield)
                # # print(no_aberration_efield[80,80])
                # no_aberration[wfs.scaledMask == 0] = numpy.nan
                
                
                
                # # temp1 = interp.zoom(phase[-1,P:Q,P:Q],wfs_size)
                # # temp1[wfs.scaledMask == 0] = numpy.nan
                
                # # plt.pcolor(xx,xx,numpy.angle(numpy.exp(1j*temp1/500.*2*3.14)).T)
                # # plt.title('aberration at altitude')
                # # plt.axis('square')
                # # plt.colorbar()
                # # plt.show()
                
                # wfs.interp_efield[wfs.scaledMask == 0] = numpy.nan
                
                # # plt.pcolor(xx,xx,numpy.angle(wfs.interp_efield.T))
                # # plt.axis('square')
                # # plt.title('actual poke phase')
                # # plt.colorbar()
                # # plt.show()
                
                # # hdu = fits.PrimaryHDU([(interp.zoom(wfs.interp_efield,128)
                # #                        / numpy.asarray(zero_wfs_efield[wfs_n])).real,
                # #                        (interp.zoom(wfs.interp_efield,128)
                # #                                               / numpy.asarray(zero_wfs_efield[wfs_n])).imag])
                # # hdul = fits.HDUList([hdu])
                # # hdul.writeto('im{:d}.fits'.format(i),overwrite=True)
                # # # hdu.close()
                
                # with_aberration_efield = (wfs.interp_efield
                #                        / numpy.asarray(zero_wfs_efield[wfs_n]))
                
                # # if i == 40:
                # #     to_plot = numpy.angle(with_aberration_efield)
                # #     to_plot -= numpy.nanmedian(to_plot)
                # #     N = with_aberration_efield.shape[0]
                # #     fig, ax1 = plt.subplots()
                    
                # #     c = ax1.pcolor(numpy.arange(N),
                # #                numpy.arange(N),
                # #                to_plot,vmax=0,vmin=-0.012)
                # #     ax1.plot([N/2 - 0.5],[N/2 - 0.5],color='r',marker='o')
                # #     fig.colorbar(c,ax=ax1)
                # #     ax1.axis('square')
                    
                # #     frame1 = ax1
                # #     for xlabel_i in frame1.axes.get_xticklabels():
                # #         xlabel_i.set_visible(False)
                # #         xlabel_i.set_fontsize(0.0)
                # #     for xlabel_i in frame1.axes.get_yticklabels():
                # #         xlabel_i.set_fontsize(0.0)
                # #         xlabel_i.set_visible(False)
                        
                # #     ax1.tick_params(axis='both', which='both', length=0)
                    
                # #     plt.gcf().set_size_inches(6,6)
                    
                # #     plt.savefig('dm_influence-' + time.strftime("%Y-%m-%d-%H-%M-%S") + '.png',
                # #                 dpi=300,bbox_inches='tight',transparent=True)
                
                # #     plt.show()
                
                # # print(numpy.nanmean(with_aberration_efield))
                # # with_aberration_efield /= numpy.nanmean(with_aberration_efield)
                # # plt.imshow(numpy.angle(with_aberration_efield))
                # # plt.show()
                # # # print(numpy.sqrt(numpy.nanmean(numpy.abs(with_aberration_efield)**2)))
                # # with_aberration_efield /= numpy.sqrt(numpy.nanmean(numpy.abs(with_aberration_efield)**2))
                # with_aberration = numpy.angle(with_aberration_efield)
                # # print(with_aberration_efield[80,80])
                # with_aberration[wfs.scaledMask == 0] = numpy.nan

                
                # # U1 = wfs.scaledMask
                # # U2 = wfs.interp_efield*wfs.scaledMask
                
                # wfs.interp_efield /= numpy.nanmean(numpy.abs(wfs.interp_efield)**2)**0.5
                
                # dm.rytov[i,wfs_n] = numpy.nanvar(numpy.log(numpy.abs(
                #     wfs.interp_efield[
                #         numpy.asarray(wfs.scaledMask,dtype=bool)])))
                
                
                # # corr = (convolve2d(numpy.nan_to_num(with_aberration),
                # #                    numpy.nan_to_num(no_aberration)))
                
                # size_A = with_aberration.shape[0]
                
                # corr = numpy.fft.fftshift(numpy.fft.ifft2(
                #     numpy.fft.fft2(numpy.pad(
                #         numpy.nan_to_num(with_aberration),
                #         ((size_A//2,size_A//2),(size_A//2,size_A//2)),mode='constant'))
                #     * numpy.conjugate(numpy.fft.fft2(numpy.pad(
                #         numpy.nan_to_num(-no_aberration),
                #         ((size_A//2,size_A//2),(size_A//2,size_A//2)),mode='constant')))
                #     )).real[size_A//2:size_A*3//2,size_A//2:size_A*3//2]
                
                # corr[wfs.scaledMask == 0] = numpy.nan
                
                # SNR = (numpy.nanmax(corr) - numpy.nanmin(corr))/numpy.nanstd(corr)
                
                # # if numpy.nanmax(corr) < numpy.abs(numpy.nanmin(corr)):
                # #     corr *= -1
                # # print(corr.shape,with_aberration.shape,no_aberration.shape)
                
                # if SNR >= 5.:
                    
                #     # MAX = numpy.nanmax(-no_aberration)
                #     # MIN = numpy.nanmin(-no_aberration)
                    
                #     # plt.pcolor(xx,xx,-no_aberration.T,vmin=MIN,vmax=MAX)
                #     # plt.axis('square')
                #     # plt.title('original poke phase')
                #     # plt.colorbar()
                #     # plt.hlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
                #     # plt.vlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
                #     # # plt.savefig('poke{:d}.png'.format(i))
                #     # plt.show()
                    
                #     # plt.pcolor(xx,xx,with_aberration.T,vmin=MIN,vmax=MAX)
                #     # plt.hlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
                #     # plt.vlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
                #     # plt.title('real interaction')
                #     # plt.axis('square')
                #     # plt.colorbar()
                #     # # plt.savefig('im{:d}.png'.format(i))
                #     # plt.show()
                
                #     x = numpy.arange(corr.shape[0],dtype=float)
                #     x -= x.max()/2.
                #     x -= 0.5
                #     x /= wfs.nx_subap_interp
                #     yy,xx = numpy.meshgrid(x,x)
                #     maxindex = numpy.array(numpy.where(corr==numpy.nanmax(corr)))[:,0]
                #     locx = xx[maxindex[0],maxindex[1]]
                #     locy = yy[maxindex[0],maxindex[1]]
                    
                #     # plt.pcolor(x,x,corr.T)#,vmin=0,vmax=1000)
                #     # plt.plot(locx,locy,ls='',marker='o',color='r')
                #     # plt.colorbar()
                #     # plt.axis('square')
                #     # plt.title('correlation map max at ({:.2f},{:.2f})'.format(locx,locy)
                #     #           + '\nSNR={:.2f}'.format(SNR))
                #     # plt.show()
                    
                #     # threshold_ratio = 3e-1
                #     # threshold = numpy.nanmax(corr)*threshold_ratio
                #     # corr[corr<threshold] = numpy.nan
                #     # corr -= threshold
                    
                    
                #     # yy,xx = numpy.meshgrid(x,x)
                    
                #     # # locx = numpy.nansum(xx*corr*wfs.scaledMask)/numpy.nansum(corr*wfs.scaledMask)
                #     # # locy = numpy.nansum(yy*corr*wfs.scaledMask)/numpy.nansum(corr*wfs.scaledMask)
                #     # locx = xx[corr==numpy.nanmax(corr)][0]
                #     # locy = yy[corr==numpy.nanmax(corr)][0]
    
                #     # if (locx == 0) or (locy == 0):
                #     #     locx = numpy.nan
                #     #     locy = numpy.nan
                        
                #     # plt.pcolor(x,x,corr.T)#,vmin=0,vmax=1000)
                #     # plt.plot(locx,locy,ls='',marker='o',color='r')
                #     # plt.colorbar()
                #     # plt.axis('square')
                #     # plt.title('correlation map max at ({:.2f},{:.2f})'.format(locx,locy))
                #     # # plt.title('correlation map with {:.2f}% threshold ({:.2f},{:.2f})'.format(threshold_ratio*100,locx,locy))
                #     # plt.show()
                    
                #     cropped_corr = corr[maxindex[0] - 1 
                #                         : maxindex[0] + 2,
                #                         maxindex[1] - 1 
                #                         : maxindex[1] + 2]
                    
                #     if ((cropped_corr[numpy.isnan(cropped_corr)].sum() <= 0.) 
                #         and (cropped_corr.size >= 9.)):
                        
                #         # print(cropped_corr)
                #         # cropped_x = numpy.arange(-1,2) / wfs.nx_subap_interp
                #         # cropped_yy,cropped_xx = numpy.meshgrid(cropped_x,cropped_x)
                        
                #         # initial_guess = numpy.array([0,0,1,1,0])
                        
                #         # params, cov = curve_fit(parabola2d,
                #         #                         (cropped_xx.flatten(),
                #         #                          cropped_yy.flatten()),
                #         #                         cropped_corr.flatten(),
                #         #                         initial_guess)
                        
                #         # true_locx = params[0] + locx
                #         # true_locy = params[1] + locy
                        
                #         # M. G. Löfdahl 2010
                        
                #         a2 = (cropped_corr[1,:].mean() - cropped_corr[-1,:].mean())/2.
                #         a3 = (cropped_corr[1,:].mean() - 2.*cropped_corr[0,:].mean() + cropped_corr[-1,:].mean())/2.
                #         a4 = (cropped_corr[:,1].mean() - cropped_corr[:,-1].mean())/2.
                #         a5 = (cropped_corr[:,1].mean() - 2.*cropped_corr[:,0].mean() + cropped_corr[:,-1].mean())/2.
                #         a6 = (cropped_corr[1,1] - cropped_corr[-1,1] - cropped_corr[1,-1] + cropped_corr[-1,-1])/4.
                        
                #         true_locx = (-1 + (2.*a2*a5 - a4*a6)/(a6**2 - 4.*a3*a5)) / wfs.nx_subap_interp  + locx
                #         true_locy = (-1 + (2.*a3*a4 - a2*a6)/(a6**2 - 4.*a3*a5)) / wfs.nx_subap_interp  + locy

                        
                #         # plt.pcolor(x,x,corr.T)#,vmin=0,vmax=1000)
                #         # plt.hlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
                #         # plt.vlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
                #         # plt.plot(locx,locy,ls='',marker='o',color='r',label='max pixel')
                #         # plt.plot(true_locx,true_locy,ls='',marker='o',color='k',label='true max')
                #         # plt.legend()
                #         # plt.colorbar()
                #         # plt.axis('square')
                #         # plt.title('correlation map max at ({:.2f},{:.2f})'.format(locx,locy)
                #         #           + '\nSNR={:.2f}'.format(SNR)
                #         #           #+ '\n x0,y0 fit = ({:.2f},{:.2f})'.format(params[0],params[1])
                #         #           + '\ntrue max at ({:.2f},{:.2f})'.format(true_locx,true_locy))
                #         # plt.show()
                        
                #         if ((numpy.abs(true_locx) < wfs.nx_subaps//2)
                #             or (numpy.abs(true_locx) < wfs.nx_subaps//2)):
                            
                #             dm.subapShift[i,wfs_n,0] = true_locx
                #             dm.subapShift[i,wfs_n,1] = true_locy
                        
                #     # else:
                #     #     true_locx = numpy.nan
                #     #     true_locy = numpy.nan
                # # else:
                # #     true_locx = numpy.nan
                # #     true_locy = numpy.nan
                
                
                
                
                # #     plt.imshow(wfs.wfsDetectorPlane)
                # #     plt.title('wfs')
                # #     plt.colorbar()
                # #     plt.show()
                
                # #wfs.slopes = wfs.slopes - zero_iMat[n_wfs_measurments: n_wfs_measurments+wfs.n_measurements]
                # # make_quiver_plot(wfs.detector_cent_coords,
                # #                  wfs.slopes
                # #                  - zero_iMat[n_wfs_measurments
                # #                              : n_wfs_measurments
                # #                              + wfs.n_measurements])
                    
                    
                
                n_wfs_measurments += wfs.n_measurements

                # Turn noise back on again if it was turned off
                if self.config.imat_noise is False:
                    wfs.config.photonNoise = wfs_pnoise
                    wfs.config.eReadNoise = wfs_rnoise

            if callback != None:
                callback()

            logger.statusMessage(i, dm.n_acts,
                                 "Generating {} Actuator DM iMat".format(dm.n_acts))
        
        if ABERRATION == True:
            logger.info("NOT Checking for redundant actuators...")
            valid_actuators = numpy.ones((dm.n_acts), dtype="int")
    
            dm.valid_actuators = valid_actuators
            n_valid_acts = valid_actuators.sum()
            logger.info("DM {} has {} valid actuators ({} dropped)".format(
                    dm.n_dm, n_valid_acts, dm.n_acts - n_valid_acts))
    
            # Can now make a final interaction matrix with only valid entries
            valid_iMat = numpy.zeros((n_valid_acts, self.sim_config.totalWfsData))
            i_valid_act = 0
            for i in range(dm.n_acts):
                if valid_actuators[i]:
                    valid_iMat[i_valid_act] = iMat[i]
                    i_valid_act += 1
    
            return valid_iMat
        
        else:
            
            logger.info("Checking for redundant actuators...")
            # Now check tath each actuator actually does something on a WFS.
            # If an act has a <0.1% effect then it will be removed
            # NOTE: THIS SHOULD REALLY BE DONE ON A PER WFS BASIS
            valid_actuators = numpy.zeros((dm.n_acts), dtype="int")
            act_threshold = abs(iMat).max() * 0.001
            for i in range(dm.n_acts):
                # plt.plot(i,abs(iMat[i]).max(),marker='o',ls='')
                if abs(iMat[i]).max() > act_threshold:
                    valid_actuators[i] = 1
                else:
                    valid_actuators[i] = 0
            # plt.hlines(act_threshold,0,dm.n_acts)
            # plt.show()
    
            dm.valid_actuators = valid_actuators
            n_valid_acts = valid_actuators.sum()
            logger.info("DM {} has {} valid actuators ({} dropped)".format(
                    dm.n_dm, n_valid_acts, dm.n_acts - n_valid_acts))
    
            # Can now make a final interaction matrix with only valid entries
            valid_iMat = numpy.zeros((n_valid_acts, self.sim_config.totalWfsData))
            i_valid_act = 0
            for i in range(dm.n_acts):
                if valid_actuators[i]:
                    valid_iMat[i_valid_act] = iMat[i]
                    i_valid_act += 1
    
            return valid_iMat


    def get_dm_imat(self, dm_index, wfs_index):
        """
        Slices and returns the interaction matrix between a given wfs and dm from teh main interaction matrix

        Parameters:
            dm_index (int): Index of required DM
            wfs_index (int): Index of required WFS

        Return:
             ndarray: interaction matrix
        """

        act_n1 = self.first_acts[dm_index]
        act_n2 = act_n1 + self.dms[dm_index].n_acts

        wfs_n1 = self.wfss[wfs_index].config.dataStart
        wfs_n2 = wfs_n1 + self.wfss[wfs_index].n_measurements
        return self.interaction_matrix[act_n1: act_n2, wfs_n1: wfs_n2]


    def makeCMat(
            self, loadIMat=True, loadCMat=True, callback=None,
            progressCallback=None):

        if loadIMat:
            try:
                self.load_interaction_matrix()
                logger.info("Interaction Matrices loaded successfully")
            except:
                tc = traceback.format_exc()
                logger.info("Load Interaction Matrices failed with error: {} - will create new one...".format(tc))
                self.makeIMat(callback=callback)
                if self.sim_config.simName is not None:
                    self.save_interaction_matrix()
                logger.info("Interaction Matrices Done")

        else:
            self.makeIMat(callback=callback)
            if self.sim_config.simName is not None:
                    self.save_interaction_matrix()
            logger.info("Interaction Matrices Done")

        if loadCMat:
            try:
                self.loadCMat()
                logger.info("Command Matrix Loaded Successfully")
            except:
                tc = traceback.format_exc()
                logger.warning("Load Command Matrix failed qith error: {} - will create new one...".format(tc))

                self.calcCMat(callback, progressCallback)
                if self.sim_config.simName is not None:
                    self.saveCMat()
                logger.info("Command Matrix Generated!")
        else:
            logger.info("Creating Command Matrix")
            self.calcCMat(callback, progressCallback)
            if self.sim_config.simName is not None:
                    self.saveCMat()
            logger.info("Command Matrix Generated!")

    def apply_gain(self):
        """
        Applies the gains set for each DM to the DM actuator commands. 
        Also applies different control law if DM is in "closed" or "open" loop mode
        """
        # Loop through DMs and apply gain
        n_act1 = int(0)
        for dm_i, dm in self.dms.items():

            n_act2 = n_act1 + int(dm.n_valid_actuators)
            # If loop is closed, only add residual measurements onto old
            # actuator values
            if dm.dmConfig.closed:
                self.actuator_values[n_act1: n_act2] += (dm.dmConfig.gain * self.new_actuator_values[n_act1: n_act2])

            else:
                self.actuator_values[n_act1: n_act2] = ((dm.dmConfig.gain * self.new_actuator_values[n_act1: n_act2])
                                + ( (1. - dm.dmConfig.gain) * self.actuator_values[n_act1: n_act2]) )

            n_act1 += int(dm.n_valid_actuators)


    def reconstruct(self, wfs_measurements):
        t = time.time()

        if self.actuator_values is None:
            self.actuator_values = numpy.zeros((int(self.sim_config.totalActs)),dtype=float)

        self.new_actuator_values = self.control_matrix.T.dot(wfs_measurements)

        self.apply_gain()

        self.Trecon += time.time()-t
        return self.actuator_values

    def reset(self):
        if self.actuator_values is not None:
            self.actuator_values[:] = 0


class MVM(Reconstructor):
    """
    Re-constructor which combines all DM interaction matrices from all DMs and
    WFSs and inverts the resulting matrix to form a global interaction matrix.
    """

    def calcCMat(self, callback=None, progressCallback=None):
        '''
        Uses DM object makeIMat methods, then inverts each to create a
        control matrix
        '''
        
        
        
        # cumulative_actuators = 0
        # old_cumulative_actuators = 0
        # for i in self.dms:
        #     if self.dms[i].dmConfig.type == 'TT':
        #         cumulative_actuators += self.dms[i].n_acts
        #         old_iMat = self.dms[i].config.iMatValue
        #         new_iMat = old_iMat*self.interaction_matrix.max()/self.interaction_matrix[old_cumulative_actuators:cumulative_actuators].max()
        #         old_cumulative_actuators = cumulative_actuators
        #         print(self.dms[i],new_iMat)
        #     # if self.dms[i].dmConfig.type =='FastPiezo':
        #     #     cumulative_actuators += self.dms[i].n_acts
        #     #     old_iMat = self.dms[i].config.iMatValue
        #     #     new_iMat = old_iMat/numpy.max(svd[old_cumulative_actuators:cumulative_actuators])
        #     #     new_iMat *= (1 - 0.2*(len(self.dms) - i)/len(self.dms))
        #     #     old_cumulative_actuators = cumulative_actuators
        #     #     print(self.dms[i],new_iMat)
        
        if self.config.svdConditioning == 'adaptive':
            rcond = get_rcond_adaptive_threshold_rank(self.interaction_matrix)
            self.config.svdConditioning = rcond
        logger.info("Invert iMat with conditioning: {:.4f}".format(
                self.config.svdConditioning))
        self.control_matrix = numpy.linalg.pinv(
                self.interaction_matrix, self.config.svdConditioning
                )
        # plt.imshow(self.interaction_matrix)
        # plt.title('control matrix')
        # plt.show()
        # _,svd,_ = numpy.linalg.svd(self.interaction_matrix)
        # svd /= svd.max()
        # plt.plot(svd,label='old svd')
        # plt.hlines(self.config.svdConditioning,0,len(svd),label='old svd')
        # plt.legend()
        # plt.show()
        # plt.imshow(self.control_matrix)
        # plt.title('control matrix')
        # plt.show()


class MVM_SeparateDMs(Reconstructor):
    """
    Re-constructor which treats a each DM Separately.

    Similar to ``MVM`` re-constructor, except each DM has its own control matrix.
    Its is assumed that each DM is "associated" with a different WFS.
    """

    def calcCMat(self,callback=None, progressCallback=None):
        '''
        Uses DM object makeIMat methods, then inverts each to create a
        control matrix
        '''
        acts = 0
        for dm_index, dm in self.dms.items():

            n_wfs_measurements = 0
            for wfs in dm.wfss:
                n_wfs_measurements += wfs.n_measurements

            dm_interaction_matrix = numpy.zeros((dm.n_acts, n_wfs_measurements))
            # Get interaction matrices from main matrix
            n_wfs_measurement = 0
            for wfs_index in [dm.dmConfig.wfs]:
                wfs = self.wfss[wfs_index]
                wfs_imat = self.get_dm_imat(dm_index, wfs_index)
                print("DM: {}, WFS: {}".format(dm_index, wfs_index))
                dm_interaction_matrix[:, n_wfs_measurement:n_wfs_measurement + wfs.n_measurements] = wfs_imat
            
            if dm.dmConfig.svdConditioning == 'adaptive':
                rcond = get_rcond_adaptive_threshold_rank(dm_interaction_matrix)
                dm.dmConfig.svdConditioning = rcond
            dm_control_matrx = numpy.linalg.pinv(dm_interaction_matrix, dm.dmConfig.svdConditioning)

            # now put carefully back into one control matrix
            for wfs_index in [dm.dmConfig.wfs]:
                wfs = self.wfss[wfs_index]
                self.control_matrix[
                        wfs.config.dataStart:
                                wfs.config.dataStart + wfs.n_measurements,
                        acts:acts+dm.n_acts] = dm_control_matrx

            acts += dm.n_acts


class LearnAndApply(MVM):
    '''
    Class to perform a simply learn and apply algorithm, where
    "learn" slopes are recorded, and an interaction matrix between off-axis
    and on-axis WFS is computed from these slopes.

    Assumes that on-axis sensor is WFS 0
    '''

    def makeIMat(self, callback=None):
        super(LearnAndApply, self).makeIMat(callback=callback)

        # only truth sensor and DM(s) interaction matrix needed 
        self.interaction_matrix = self.interaction_matrix[:,:2*self.wfss[0].n_subaps]

    def saveCMat(self):
        cMatFilename = self.sim_config.simName+"/cMat.fits"
        tomoMatFilename = self.sim_config.simName+"/tomoMat.fits"

        fits.writeto(
                cMatFilename, self.control_matrix,
                header=self.sim_config.saveHeader, overwrite=True
                )

        fits.writeto(
                tomoMatFilename, self.tomoRecon,
                header=self.sim_config.saveHeader, overwrite=True
                )

    def loadCMat(self):

        super(LearnAndApply, self).loadCMat()

        #Load tomo reconstructor
        tomoFilename = self.sim_config.simName+"/tomoMat.fits"
        tomoMat = fits.getdata(tomoFilename)

        #And check its the right size
        if tomoMat.shape != (
                2*self.wfss[0].n_subaps,
                self.sim_config.totalWfsData - 2*self.wfss[0].n_subaps):
            logger.warning("Loaded Tomo matrix not the expected shape - gonna make a new one..." )
            raise Exception
        else:
            self.tomoRecon = tomoMat


    def initControlMatrix(self):

        self.controlShape = (2*self.wfss[0].n_subaps, self.sim_config.totalActs)
        self.control_matrix = numpy.zeros( self.controlShape )


    def learn(self, callback=None, progressCallback=None):
        '''
        Takes "self.learnFrames" WFS frames, and computes the tomographic
        reconstructor for the system. This method uses the "truth" sensor, and
        assumes that this is WFS0
        '''

        self.learnSlopes = numpy.zeros( (self.learnIters,self.sim_config.totalWfsData) )
        for i in xrange(self.learnIters):
            self.learnIter=i

            scrns = self.moveScrns()

            for j in range(len(self.wfss)):
                wfs = self.wfss[j]
                self.learnSlopes[i,j*wfs.n_measurements:(j+1)*wfs.n_measurements] = wfs.frame(scrns, read=True)


            logger.statusMessage(i+1, self.learnIters, "Performing Learn")
            if callback!=None:
                callback()
            if progressCallback!=None:
               progressCallback("Performing Learn", i, self.learnIters )

        if self.sim_config.saveLearn:
            #FITS.Write(self.learnSlopes,self.sim_config.simName+"/learn.fits")
            fits.writeto(
                    self.sim_config.simName+"/learn.fits",
                    self.learnSlopes, header=self.sim_config.saveHeader,
                    overwrite=True )


    def calcCMat(self,callback=None, progressCallback=None):
        '''
        Uses the slopes recorded in the "learn" and DM interaction matrices
        to create a CMat.
        '''

        logger.info("Performing Learn....")
        self.learn(callback, progressCallback)
        logger.info("Done. Creating Tomographic Reconstructor...")

        if progressCallback!=None:
            progressCallback(1,1, "Calculating Covariance Matrices")

        self.covMat = numpy.cov(self.learnSlopes.T)
        Conoff = self.covMat[   :2*self.wfss[0].n_subaps,
                                2*self.wfss[0].n_subaps:     ]
        Coffoff = self.covMat[  2*self.wfss[0].n_subaps:,
                                2*self.wfss[0].n_subaps:    ]

        logger.info("Inverting offoff Covariance Matrix")
        iCoffoff = numpy.linalg.pinv(Coffoff, rcond=1e-8)

        self.tomoRecon = Conoff.dot(iCoffoff)
        logger.info("Done. \nCreating full reconstructor....")

        #Same code as in "MVM" class to create dm-slopes reconstructor.

        super(LearnAndApply, self).calcCMat(callback, progressCallback)

        #Dont make global reconstructor. Will reconstruct on-axis slopes, then
        #dmcommands explicitly
        #self.controlMatrix = (self.controlMatrix.T.dot(self.tomoRecon)).T
        logger.info("Done.")


    def reconstruct(self, slopes):
        """
        Determine DM commands using previously made
        reconstructor from slopes.
        Args:
            slopes (ndarray): array of slopes to reconstruct from
        Returns:
            ndarray: array of commands to be sent to DM
        """

        #Retreive pseudo on-axis slopes from tomo reconstructor
        slopes = self.tomoRecon.dot(slopes[2*self.wfss[0].n_subaps:])

        if self.dms[0].dmConfig.type=="TT":
            ttMean = slopes.reshape(2, self.wfss[0].n_subaps).mean(1)
            ttCommands = self.control_matrix[:,:2].T.dot(slopes)
            slopes[:self.wfss[0].n_subaps] -= ttMean[0]
            slopes[self.wfss[0].n_subaps:] -= ttMean[1]

            #get dm commands for the calculated on axis slopes
            dmCommands = self.control_matrix[:,2:].T.dot(slopes)

            return numpy.append(ttCommands, dmCommands)

        #get dm commands for the calculated on axis slopes
        dmCommands = super(LearnAndApply, self).reconstruct(slopes)
        #dmCommands = self.control_matrix.T.dot(slopes)
        return dmCommands


class LearnAndApplyLTAO(LearnAndApply, MVM_SeparateDMs):
    '''
    Class to perform a simply learn and apply algorithm, where
    "learn" slopes are recorded, and an interaction matrix between off-axis
    and on-axis WFS is computed from these slopes.

    This is an ``
    Assumes that on-axis sensor is WFS 1
    '''

    def initcontrol_matrix(self):

        self.controlShape = (2*(self.wfss[0].activeSubaps+self.wfss[1].activeSubaps), self.sim_config.totalActs)
        self.control_matrix = numpy.zeros( self.controlShape )


    def calcCMat(self,callback=None, progressCallback=None):
        '''
        Uses the slopes recorded in the "learn" and DM interaction matrices
        to create a CMat.
        '''

        logger.info("Performing Learn....")
        self.learn(callback, progressCallback)
        logger.info("Done. Creating Tomographic Reconstructor...")

        if progressCallback!=None:
            progressCallback(1,1, "Calculating Covariance Matrices")

        self.covMat = numpy.cov(self.learnSlopes.T)
        Conoff = self.covMat[
                self.wfss[1].config.dataStart:
                        self.wfss[2].config.dataStart,
                self.wfss[2].config.dataStart:
                ]
        Coffoff = self.covMat[  self.wfss[2].config.dataStart:,
                                self.wfss[2].config.dataStart:    ]

        logger.info("Inverting offoff Covariance Matrix")
        iCoffoff = numpy.linalg.pinv(Coffoff)

        self.tomoRecon = Conoff.dot(iCoffoff)
        logger.info("Done. \nCreating full reconstructor....")

        #Same code as in "MVM" class to create dm-slopes reconstructor.

        MVM_SeparateDMs.calcCMat(self, callback, progressCallback)

        #Dont make global reconstructor. Will reconstruct on-axis slopes, then
        #dmcommands explicitly
        #self.control_matrix = (self.control_matrix.T.dot(self.tomoRecon)).T
        logger.info("Done.")

    def reconstruct(self, slopes):
        """
        Determine DM commands using previously made
        reconstructor from slopes.
        Args:
            slopes (ndarray): array of slopes to reconstruct from
        Returns:
            ndarray: array to comands to be sent to DM
        """

        #Retreive pseudo on-axis slopes from tomo reconstructor
        slopes_HO = self.tomoRecon.dot(
                slopes[self.wfss[2].config.dataStart:])

        # Probably should remove TT from these slopes?
        nSubaps = slopes_HO.shape[0]
        slopes_HO[:nSubaps] -= slopes_HO[:nSubaps].mean()
        slopes_HO[nSubaps:] -= slopes_HO[nSubaps:].mean()

        # Final slopes are TT slopes appended to the tomographic High order slopes
        onSlopes = numpy.append(
                slopes[:self.wfss[1].config.dataStart], slopes_HO)

        dmCommands = self.control_matrix.T.dot(onSlopes)

        #
        # ttCommands = self.control_matrix[
        #         :self.wfss[1].config.dataStart,:2].T.dot(slopes_TT)
        #
        # hoCommands = self.control_matrix[
        #         self.wfss[1].config.dataStart:,2:].T.dot(slopes_HO)
        #
        # #if self.dms[0].dmConfig.type=="TT":
        #    ttMean = slopes.reshape(2, self.wfss[0].activeSubaps).mean(1)
        #    ttCommands = self.control_matrix[:,:2].T.dot(slopes)
        #    slopes[:self.wfss[0].activeSubaps] -= ttMean[0]
        #    slopes[self.wfss[0].activeSubaps:] -= ttMean[1]

        #    #get dm commands for the calculated on axis slopes
        #    dmCommands = self.control_matrix[:,2:].T.dot(slopes)

        #    return numpy.append(ttCommands, dmCommands)

        #get dm commands for the calculated on axis slopes

       # dmCommands = self.control_matrix.T.dot(slopes)

        return dmCommands



#####################################
#Experimental....
#####################################
class GLAO_4LGS(MVM):
    """
    Reconstructor of LGS TT prediction algorithm.

    Uses one TT DM and a high order DM. The TT WFS controls the TT DM and
    the second WFS controls the high order DM. The TT WFS and DM are
    assumed to be the first in the system.
    """


    def initControlMatrix(self):

        self.controlShape = (2*self.wfss[0].activeSubaps+2*self.wfss[1].activeSubaps,
                             self.sim_config.totalActs)
        self.controlMatrix = numpy.zeros( self.controlShape )


    def reconstruct(self, slopes):
        """
        Determine DM commands using previously made
        reconstructor from slopes.
        Args:
            slopes (ndarray): array of slopes to reconstruct from
        Returns:
            ndarray: array to commands to be sent to DM
        """

        offSlopes = slopes[self.wfss[2].config.dataStart:]
        meanOffSlopes = offSlopes.reshape(4,self.wfss[2].activeSubaps*2).mean(0)

        meanOffSlopes = self.removeCommonTT(meanOffSlopes, [1])

        slopes = numpy.append(
                slopes[:self.wfss[1].config.dataStart], meanOffSlopes)

        return super(LgsTT, self).reconstruct(slopes)


    def removeCommonTT(self, slopes, wfsList):

        xSlopesShape = numpy.array(slopes.shape)
        xSlopesShape[-1] /= 2.
        xSlopes = numpy.zeros(xSlopesShape)
        ySlopes = numpy.zeros(xSlopesShape)

        for i in range(len(wfsList)):
            wfs = wfsList[i]
            wfsSubaps = self.wfss[wfs].activeSubaps
            xSlopes[..., i*wfsSubaps:(i+1)*wfsSubaps] = slopes[..., i*2*wfsSubaps:i*2*wfsSubaps+wfsSubaps]
            ySlopes[..., i*wfsSubaps:(i+1)*wfsSubaps] = slopes[..., i*2*wfsSubaps+wfsSubaps:i*2*wfsSubaps+2*wfsSubaps]

        xSlopes = (xSlopes.T - xSlopes.mean(-1)).T
        ySlopes = (ySlopes.T - ySlopes.mean(-1)).T

        for i in range(len(wfsList)):
            wfs = wfsList[i]
            wfsSubaps = self.wfss[wfs].activeSubaps

            slopes[..., i*2*wfsSubaps:i*2*wfsSubaps+wfsSubaps] = xSlopes[..., i*wfsSubaps:(i+1)*wfsSubaps]
            slopes[..., i*2*wfsSubaps+wfsSubaps:i*2*wfsSubaps+2*wfsSubaps] = ySlopes[..., i*wfsSubaps:(i+1)*wfsSubaps]

        return slopes

class WooferTweeter(Reconstructor):
    '''
    Reconstructs a 2 DM system, where 1 DM is of low order, high stroke
    and the other has a higher, but low stroke.

    Reconstructs dm commands for each DM, then removes the low order
    component from the high order commands by propagating back to the
    slopes corresponding to the lower order DM shape, and propagating
    to the high order DM shape.
    '''

    def calcCMat(self,callback=None, progressCallback=None):
        '''
        Creates control Matrix.
        Assumes that DM 0  is low order,
        and DM 1 is high order.
        '''

        if self.sim_config.nDM==1:
            logger.warning("Woofer Tweeter Reconstruction not valid for 1 dm.")
            return None
        acts = 0
        dmCMats = []
        for dm in xrange(self.sim_config.nDM):
            dmIMat = self.dms[dm].iMat
            
            if self.dms[dm].dmConfig.svdConditioning == 'adaptive':
                rcond = get_rcond_adaptive_threshold_rank(dmIMat)
                self.dms[dm].dmConfig.svdConditioning = rcond
            
            logger.info("Invert DM {} IMat with conditioning:{}".format(dm,self.dms[dm].dmConfig.svdConditioning))
            if dmIMat.shape[0]==dmIMat.shape[1]:
                dmCMat = numpy.linalg.pinv(dmIMat)
            else:
                dmCMat = numpy.linalg.pinv(
                                    dmIMat, self.dms[dm].dmConfig.svdConditioning)

            #if dm != self.sim_config.nDM-1:
            #    self.controlMatrix[:,acts:acts+self.dms[dm].n_acts] = dmCMat
            #    acts+=self.dms[dm].n_acts

            dmCMats.append(dmCMat)


        self.control_matrix[:, 0:self.dms[0].n_acts]
        acts = self.dms[0].n_acts
        for dm in range(1, self.sim_config.nDM):

            #This is the matrix which converts from Low order DM commands
            #to high order DM commands, via slopes
            lowToHighTransform = self.dms[dm-1].iMat.T.dot( dmCMats[dm-1] )

            highOrderCMat = dmCMats[dm].T.dot(
                    numpy.identity(self.sim_config.totalWfsData)-lowToHighTransform)

            dmCMats[dm] = highOrderCMat

            self.control_matrix[:, acts:acts + self.dms[dm].n_acts] = highOrderCMat.T
            acts += self.dms[dm].n_acts


class LgsTT(LearnAndApply):
    """
    Reconstructor of LGS TT prediction algorithm.

    Uses one TT DM and a high order DM. The TT WFS controls the TT DM and
    the second WFS controls the high order DM. The TT WFS and DM are
    assumed to be the first in the system.
    """

    def initControlMatrix(self):

        self.controlShape = (2*self.wfss[0].activeSubaps+2*self.wfss[1].activeSubaps,
                             self.sim_config.totalActs)
        self.controlMatrix = numpy.zeros( self.controlShape )


    def calcCMat(self,callback=None, progressCallback=None):
        '''
        Uses the slopes recorded in the "learn" and DM interaction matrices
        to create a CMat.
        '''

        logger.info("Performing Learn....")
        self.learn(callback, progressCallback)
        logger.info("Done. Creating Tomographic Reconstructor...")

        if progressCallback!=None:
            progressCallback(1,1, "Calculating Covariance Matrices")

        #Need to remove all *common* TT from off-axis learn slopes
        self.learnSlopes[:, 2*self.wfss[1].activeSubaps:] = self.removeCommonTT(
                self.learnSlopes[:, 2*self.wfss[1].activeSubaps:], [2,3,4,5])

        self.covMat = numpy.cov(self.learnSlopes.T)
        Conoff = self.covMat[   :2*self.wfss[1].activeSubaps,
                                2*self.wfss[1].activeSubaps:     ]
        Coffoff = self.covMat[  2*self.wfss[1].activeSubaps:,
                                2*self.wfss[1].activeSubaps:    ]

        logger.info("Inverting offoff Covariance Matrix")
        iCoffoff = numpy.linalg.pinv(Coffoff)

        self.tomoRecon = Conoff.dot(iCoffoff)
        logger.info("Done. \nCreating full reconstructor....")

        super(LgsTT, self).calcCMat(callback, progressCallback)


    def reconstruct(self, slopes):
        """
        Determine DM commands using previously made
        reconstructor from slopes.
        Args:
            slopes (ndarray): array of slopes to reconstruct from
        Returns:
            ndarray: array to commands to be sent to DM
        """

        #Get off axis slopes and remove *common* TT
        offSlopes = slopes[self.wfss[2].config.dataStart:]
        offSlopes = self.removeCommonTT(offSlopes,[2,3,4,5])

        #Use the tomo matrix to get pseudo on-axis slopes
        psuedoOnSlopes = self.tomoRecon.dot(offSlopes)

        #Combine on-axis slopes with TT measurements
        slopes = numpy.append(
                slopes[:self.wfss[1].config.dataStart], psuedoOnSlopes)

        #Send to command matrices to get dmCommands
        return super(LgsTT, self).reconstruct(slopes)


    def removeCommonTT(self, slopes, wfsList):

        xSlopesShape = numpy.array(slopes.shape)
        xSlopesShape[-1] /= 2.
        xSlopes = numpy.zeros(xSlopesShape)
        ySlopes = numpy.zeros(xSlopesShape)

        for i in range(len(wfsList)):
            wfs = wfsList[i]
            wfsSubaps = self.wfss[wfs].activeSubaps
            xSlopes[..., i*wfsSubaps:(i+1)*wfsSubaps] = slopes[..., i*2*wfsSubaps:i*2*wfsSubaps+wfsSubaps]
            ySlopes[..., i*wfsSubaps:(i+1)*wfsSubaps] = slopes[..., i*2*wfsSubaps+wfsSubaps:i*2*wfsSubaps+2*wfsSubaps]

        xSlopes = (xSlopes.T - xSlopes.mean(-1)).T
        ySlopes = (ySlopes.T - ySlopes.mean(-1)).T

        for i in range(len(wfsList)):
            wfs = wfsList[i]
            wfsSubaps = self.wfss[wfs].activeSubaps

            slopes[..., i*2*wfsSubaps:i*2*wfsSubaps+wfsSubaps] = xSlopes[..., i*wfsSubaps:(i+1)*wfsSubaps]
            slopes[..., i*2*wfsSubaps+wfsSubaps:i*2*wfsSubaps+2*wfsSubaps] = ySlopes[..., i*wfsSubaps:(i+1)*wfsSubaps]

        return slopes

class ANN(Reconstructor):
    """
    Reconstructs using a neural net
    Assumes on axis slopes are WFS 0

    Net must be set by setting ``sim.recon.net = net`` before loop is run
    net object must have a ``run`` method, which accepts slopes and returns
    on Axis slopes
    """

    def calcCMat(self, callback=None, progressCallback=None):

        nSlopes = self.wfss[0].activeSubaps*2

        self.controlShape = (nSlopes, self.sim_config.totalActs)
        self.controlMatrix = numpy.zeros((nSlopes, self.sim_config.totalActs))
        acts = 0
        for dm in xrange(self.sim_config.nDM):
            dmIMat = self.dms[dm].iMat

            if dmIMat.shape[0]==dmIMat.shape[1]:
                dmCMat = numpy.inv(dmIMat)
            else:
                dmCMat = numpy.linalg.pinv(dmIMat, self.dmConds[dm])

            self.controlMatrix[:,acts:acts+self.dms[dm].n_acts] = dmCMat
            acts += self.dms[dm].n_acts

    def reconstruct(self, slopes):
        """
        Determine DM commands using previously made
        reconstructor from slopes. Uses Artificial Neural Network.

        Slopes are normalised before being run through the network.

        Args:
            slopes (ndarray): array of slopes to reconstruct from
        Returns:
            ndarray: array to comands to be sent to DM
        """
        t=time.time()
        offSlopes = slopes[self.wfss[0].activeSubaps*2:]/7 # normalise
        onSlopes = self.net.run(offSlopes)*7 # un-normalise
        dmCommands = self.controlMatrix.T.dot(onSlopes)

        self.Trecon += time.time()-t
        return dmCommands

def get_rcond_adaptive_threshold_rank(A,plot=False,return_place=False):
  # _,s,_ = numpy.linalg.svd(A)
  # s /= s.max()
  # energy = s**2
  # energy /= energy.sum()
  # accumulated_energy = numpy.zeros_like(energy)
  # threshold = 1/2./numpy.linalg.matrix_rank(A)

  # for i in numpy.arange(energy.shape[0]):
  #   accumulated_energy[i] = energy[:i].sum()
  # residual = 1 - accumulated_energy
  # place = numpy.where(residual<=threshold)[0][0]
  
  # if plot==True:
  #   # plt.plot(residual,label='residual')
  #   # plt.hlines(residual[place],0,s.shape[0])
  #   # plt.vlines(place,0,1)
  #   # plt.yscale('log')
  #   # plt.show()

  #   plt.plot(s)
  #   plt.hlines(s[place],0,s.shape[0])
  #   plt.vlines(place,0,1)
  #   plt.yscale('log')
  #   plt.show()
  # rcond = s[place]
  # if return_place == True:
  #   return rcond, place
  # else:
  #   return rcond
  return 0.05

def pinv_adaptive_threshold_rank(A,return_rcond=False):
    if return_rcond == False:
        rcond = get_rcond_adaptive_threshold_rank(A)
        pinvA = numpy.linalg.pinv(A,rcond=rcond)
        return pinvA
    else:
        rcond, place = get_rcond_adaptive_threshold_rank(A,return_place=True)
        return pinvA, rcond, place

# def make_quiver_plot(wfs):
#     position = wfs.detector_cent_coords
#     N = position.shape[0]
    
#     step = (position[N//2 + 1,1]
#             - position[N//2,1])
#     position = position // step
    
#     slopex = wfs.slopes[:N]
#     slopey = wfs.slopes[N:]
#     plt.quiver(rearrange1(slopex, position),
#                 rearrange1(slopey, position),
#                 scale=5, scale_units='inches')
#     plt.axis('square')
#     plt.show()
#     return

def make_quiver_plot(position, slopes):
    #position = wfs.detector_cent_coords
    N = position.shape[0]
    
    step = (position[N//2 + 1,1]
            - position[N//2,1])
    position = position // step
    
    max_position = position.max()
    x = numpy.arange(max_position + 1.)
    x -= max_position/2.
    yy, xx = numpy.meshgrid(x,x)
    
    slopex = slopes[:N]#wfs.slopes[:N]
    slopey = slopes[N:]#wfs.slopes[N:]
    plt.quiver(xx,-yy,
               rearrange1(slopex, position).T,
               rearrange1(-slopey, position).T,
               scale=5, scale_units='inches')
    plt.title('IM Slope')
    plt.axis('square')
    plt.show()
    return

def rearrange1(A, position):
    """
    rearrange soapy reported wfs value from 1d with skips into 2d

    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    position : TYPE
        DESCRIPTION.

    Returns
    -------
    a : TYPE
        DESCRIPTION.

    """
    N = position.shape[0]
    size = numpy.max(position) - numpy.min(position) + 1
    DIM = position.shape[1]
    a = numpy.zeros((size,)*DIM, dtype=float)
    for n in numpy.arange(N):
        a[tuple(position[n])] = A[n]
    return a

def parabola2d(xy,x0,y0,ax,ay,c):
    x, y = xy
    z = ax*(x - x0)**2 + ay*(y - y0)**2 + c
    return z

# iMat = poke(dm,self.n_dms,self.scrn_size,self.dms,self.wfss,self.config.imat_noise,zero_iMat,iMat,callback)
# @numba.jit(nopython=False, parallel=True)
# def poke(dm,n_dms,scrn_size,dms,wfss,imat_noise,zero_iMat,iMat,callback):
#     for i in range(dm.n_acts):
#         actCommands = numpy.zeros(dm.n_acts)
#         # Set vector of iMat commands and phase to 0
#         actCommands[:] = 0

#         # Except the one we want to make an iMat for!
#         actCommands[i] = 1 # dm.dmConfig.iMatValue
#         phase = numpy.zeros((n_dms, scrn_size, scrn_size))
#         # Now get a DM shape for that command
#         phase[dm.n_dm] = dm.dmFrame(actCommands)
#         for DM_N, DM in dms.items():
#             if DM.config.type == 'Aberration':
#                 if DM.config.calibrate == False:
#                     phase[DM_N] = DM.dmFrame(1)
#                 else:
#                     phase[DM_N] = DM.dmFrame(0)
#         # Send the DM shape off to the relavent WFS. put result in iMat
#         n_wfs_measurments = 0
#         for wfs_n, wfs in wfss.items():
#             # turn off wfs noise if set
#             if imat_noise is False:
#                 wfs_pnoise = wfs.config.photonNoise
#                 wfs.config.photonNoise = False
#                 wfs_rnoise = wfs.config.eReadNoise
#                 wfs.config.eReadNoise = 0
            
#             # plot = False
#             # if (plot == True) and (i == 40):
#             # #     plt.imshow(phase.sum(0))
#             # #     plt.title('all geom dm')
#             # #     plt.colorbar()
#             # #     plt.show()
#             #     actCommands = numpy.zeros((81),dtype=float)
#             #     actCommands[20] = 1
#             #     actCommands[22] = 1
#             #     actCommands[24] = 1
#             #     actCommands[38] = 1
#             #     actCommands[40] = 1
#             #     actCommands[42] = 1
#             #     actCommands[56] = 1
#             #     actCommands[58] = 1
#             #     actCommands[60] = 1
#             #     phase[dm.n_dm] = dm.dmFrame(actCommands)
#             #     wfs.frame(None, phase_correction=phase, iMatFrame=True)
                
#             #     xx = numpy.arange(numpy.array(zero_wfs_efield[wfs_n]).shape[0],dtype=float)
#             #     xx -= xx.max()/2.
#             #     xx /= wfs.nx_subap_interp
                
#             #     zero_wfs_efield[wfs_n][wfs.scaledMask == 0] = numpy.nan
                
                
#             #     wfs_size = wfs.interp_efield.shape[0]
                
#             #     A = phase.shape[1]
#             #     B = self.soapy_config.sim.pupilSize
#             #     P = (A - B)//2
#             #     Q = (A + B)//2
                
#             #     no_aberration_efield = (numpy.exp(1j*interp.zoom(phase[:-1,P:Q,P:Q].sum(0),wfs_size)/500.*2*numpy.pi))
                
#             #     wfs.interp_efield[wfs.scaledMask == 0] = numpy.nan
                
#             #     with_aberration_efield = (wfs.interp_efield
#             #                             / numpy.asarray(zero_wfs_efield[wfs_n]))
                
#             #     to_plot = numpy.angle(with_aberration_efield)
#             #     to_plot -= numpy.nanmedian(to_plot)
#             #     fig, ax1 = plt.subplots()
                
#             #     c = ax1.pcolor(xx,xx,
#             #                 -to_plot,vmin=0,vmax=0.012)
                
#             #     ax1.hlines((-4,-3,-2,-1,0,1,2,3,4),xmin=-4,xmax=4,color='w')
#             #     ax1.vlines((-4,-3,-2,-1,0,1,2,3,4),ymin=-4,ymax=4,color='w')
#             #     ax1.plot([0,2,-2,0,0,-2,-2,2,2],[0,0,0,2,-2,-2,2,-2,2],color='k',marker='o',ls='')
#             #     fig.colorbar(c,ax=ax1)
#             #     ax1.axis('square')
                
#             #     frame1 = ax1
#             #     for xlabel_i in frame1.axes.get_xticklabels():
#             #         xlabel_i.set_visible(False)
#             #         xlabel_i.set_fontsize(0.0)
#             #     for xlabel_i in frame1.axes.get_yticklabels():
#             #         xlabel_i.set_fontsize(0.0)
#             #         xlabel_i.set_visible(False)
                    
#             #     ax1.tick_params(axis='both', which='both', length=0)
                
#             #     plt.gcf().set_size_inches(6,6)
                
#             #     plt.savefig('dm_influence-' + time.strftime("%Y-%m-%d-%H-%M-%S") + '.png',
#             #                 dpi=300,bbox_inches='tight',transparent=True)
            
#             #     plt.show()
            
#             iMat[i, n_wfs_measurments: n_wfs_measurments+wfs.n_measurements] = -1 * (
#                 wfs.frame(None, phase_correction=phase, iMatFrame=True)
#                 - zero_iMat[n_wfs_measurments: n_wfs_measurments+wfs.n_measurements])# / dm.dmConfig.iMatValue
#             # # plt.plot(wfs.frame(None, phase_correction=phase, iMatFrame=True),label='wfs measurement')
#             # # plt.plot(zero_iMat[n_wfs_measurments: n_wfs_measurments+wfs.n_measurements],label='zero')
#             # # plt.plot(iMat[i, n_wfs_measurments: n_wfs_measurments+wfs.n_measurements],label='IM')
#             # # plt.legend()
#             # # plt.show()
            
#             # # if i == 40:
#             # #     plt.imshow(phase.sum(0))
#             # #     plt.title('all geom dm')
#             # #     plt.colorbar()
#             # #     plt.show()
            
#             # xx = numpy.arange(numpy.array(zero_wfs_efield[wfs_n]).shape[0],dtype=float)
#             # xx -= xx.max()/2.
#             # xx /= wfs.nx_subap_interp
            
#             # zero_wfs_efield[wfs_n][wfs.scaledMask == 0] = numpy.nan
            
#             # # plt.pcolor(xx,xx,numpy.angle(numpy.asarray(zero_wfs_efield[wfs_n])).T)
#             # # plt.axis('square')
#             # # plt.title('0 phase')
#             # # plt.colorbar()
#             # # plt.show()
            
#             # # hdu = fits.PrimaryHDU([numpy.exp(1j*phase[:-1,129:257,129:257].sum(0)/500.*2*3.14).real,
#             # #                       numpy.exp(1j*phase[:-1,129:257,129:257].sum(0)/500.*2*3.14).imag])
#             # # hdul = fits.HDUList([hdu])
#             # # hdul.writeto('poke{:d}.fits'.format(i),overwrite=True)
#             # # # hdu.close()
            
#             # wfs_size = wfs.interp_efield.shape[0]
            
#             # A = phase.shape[1]
#             # B = self.soapy_config.sim.pupilSize
#             # P = (A - B)//2
#             # Q = (A + B)//2
            
#             # no_aberration_efield = (numpy.exp(1j*interp.zoom(phase[:-1,P:Q,P:Q].sum(0),wfs_size)/500.*2*numpy.pi))
#             # min_phase = -0.012#(-numpy.angle(no_aberration_efield)).min()
#             # max_phase = 0#(-numpy.angle(no_aberration_efield)).max()
#             # # no_aberration_efield /= numpy.nanmean(no_aberration_efield)
#             # # no_aberration_efield /= numpy.sqrt(numpy.nanmean(numpy.abs(no_aberration_efield)**2))
#             # no_aberration = numpy.angle(no_aberration_efield)
#             # # print(no_aberration_efield[80,80])
#             # no_aberration[wfs.scaledMask == 0] = numpy.nan
            
            
            
#             # # temp1 = interp.zoom(phase[-1,P:Q,P:Q],wfs_size)
#             # # temp1[wfs.scaledMask == 0] = numpy.nan
            
#             # # plt.pcolor(xx,xx,numpy.angle(numpy.exp(1j*temp1/500.*2*3.14)).T)
#             # # plt.title('aberration at altitude')
#             # # plt.axis('square')
#             # # plt.colorbar()
#             # # plt.show()
            
#             # wfs.interp_efield[wfs.scaledMask == 0] = numpy.nan
            
#             # # plt.pcolor(xx,xx,numpy.angle(wfs.interp_efield.T))
#             # # plt.axis('square')
#             # # plt.title('actual poke phase')
#             # # plt.colorbar()
#             # # plt.show()
            
#             # # hdu = fits.PrimaryHDU([(interp.zoom(wfs.interp_efield,128)
#             # #                        / numpy.asarray(zero_wfs_efield[wfs_n])).real,
#             # #                        (interp.zoom(wfs.interp_efield,128)
#             # #                                               / numpy.asarray(zero_wfs_efield[wfs_n])).imag])
#             # # hdul = fits.HDUList([hdu])
#             # # hdul.writeto('im{:d}.fits'.format(i),overwrite=True)
#             # # # hdu.close()
            
#             # with_aberration_efield = (wfs.interp_efield
#             #                        / numpy.asarray(zero_wfs_efield[wfs_n]))
            
#             # # if i == 40:
#             # #     to_plot = numpy.angle(with_aberration_efield)
#             # #     to_plot -= numpy.nanmedian(to_plot)
#             # #     N = with_aberration_efield.shape[0]
#             # #     fig, ax1 = plt.subplots()
                
#             # #     c = ax1.pcolor(numpy.arange(N),
#             # #                numpy.arange(N),
#             # #                to_plot,vmax=0,vmin=-0.012)
#             # #     ax1.plot([N/2 - 0.5],[N/2 - 0.5],color='r',marker='o')
#             # #     fig.colorbar(c,ax=ax1)
#             # #     ax1.axis('square')
                
#             # #     frame1 = ax1
#             # #     for xlabel_i in frame1.axes.get_xticklabels():
#             # #         xlabel_i.set_visible(False)
#             # #         xlabel_i.set_fontsize(0.0)
#             # #     for xlabel_i in frame1.axes.get_yticklabels():
#             # #         xlabel_i.set_fontsize(0.0)
#             # #         xlabel_i.set_visible(False)
                    
#             # #     ax1.tick_params(axis='both', which='both', length=0)
                
#             # #     plt.gcf().set_size_inches(6,6)
                
#             # #     plt.savefig('dm_influence-' + time.strftime("%Y-%m-%d-%H-%M-%S") + '.png',
#             # #                 dpi=300,bbox_inches='tight',transparent=True)
            
#             # #     plt.show()
            
#             # # print(numpy.nanmean(with_aberration_efield))
#             # # with_aberration_efield /= numpy.nanmean(with_aberration_efield)
#             # # plt.imshow(numpy.angle(with_aberration_efield))
#             # # plt.show()
#             # # # print(numpy.sqrt(numpy.nanmean(numpy.abs(with_aberration_efield)**2)))
#             # # with_aberration_efield /= numpy.sqrt(numpy.nanmean(numpy.abs(with_aberration_efield)**2))
#             # with_aberration = numpy.angle(with_aberration_efield)
#             # # print(with_aberration_efield[80,80])
#             # with_aberration[wfs.scaledMask == 0] = numpy.nan

            
#             # # U1 = wfs.scaledMask
#             # # U2 = wfs.interp_efield*wfs.scaledMask
            
#             # wfs.interp_efield /= numpy.nanmean(numpy.abs(wfs.interp_efield)**2)**0.5
            
#             # dm.rytov[i,wfs_n] = numpy.nanvar(numpy.log(numpy.abs(
#             #     wfs.interp_efield[
#             #         numpy.asarray(wfs.scaledMask,dtype=bool)])))
            
            
#             # # corr = (convolve2d(numpy.nan_to_num(with_aberration),
#             # #                    numpy.nan_to_num(no_aberration)))
            
#             # size_A = with_aberration.shape[0]
            
#             # corr = numpy.fft.fftshift(numpy.fft.ifft2(
#             #     numpy.fft.fft2(numpy.pad(
#             #         numpy.nan_to_num(with_aberration),
#             #         ((size_A//2,size_A//2),(size_A//2,size_A//2)),mode='constant'))
#             #     * numpy.conjugate(numpy.fft.fft2(numpy.pad(
#             #         numpy.nan_to_num(-no_aberration),
#             #         ((size_A//2,size_A//2),(size_A//2,size_A//2)),mode='constant')))
#             #     )).real[size_A//2:size_A*3//2,size_A//2:size_A*3//2]
            
#             # corr[wfs.scaledMask == 0] = numpy.nan
            
#             # SNR = (numpy.nanmax(corr) - numpy.nanmin(corr))/numpy.nanstd(corr)
            
#             # # if numpy.nanmax(corr) < numpy.abs(numpy.nanmin(corr)):
#             # #     corr *= -1
#             # # print(corr.shape,with_aberration.shape,no_aberration.shape)
            
#             # if SNR >= 5.:
                
#             #     # MAX = numpy.nanmax(-no_aberration)
#             #     # MIN = numpy.nanmin(-no_aberration)
                
#             #     # plt.pcolor(xx,xx,-no_aberration.T,vmin=MIN,vmax=MAX)
#             #     # plt.axis('square')
#             #     # plt.title('original poke phase')
#             #     # plt.colorbar()
#             #     # plt.hlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
#             #     # plt.vlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
#             #     # # plt.savefig('poke{:d}.png'.format(i))
#             #     # plt.show()
                
#             #     # plt.pcolor(xx,xx,with_aberration.T,vmin=MIN,vmax=MAX)
#             #     # plt.hlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
#             #     # plt.vlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
#             #     # plt.title('real interaction')
#             #     # plt.axis('square')
#             #     # plt.colorbar()
#             #     # # plt.savefig('im{:d}.png'.format(i))
#             #     # plt.show()
            
#             #     x = numpy.arange(corr.shape[0],dtype=float)
#             #     x -= x.max()/2.
#             #     x -= 0.5
#             #     x /= wfs.nx_subap_interp
#             #     yy,xx = numpy.meshgrid(x,x)
#             #     maxindex = numpy.array(numpy.where(corr==numpy.nanmax(corr)))[:,0]
#             #     locx = xx[maxindex[0],maxindex[1]]
#             #     locy = yy[maxindex[0],maxindex[1]]
                
#             #     # plt.pcolor(x,x,corr.T)#,vmin=0,vmax=1000)
#             #     # plt.plot(locx,locy,ls='',marker='o',color='r')
#             #     # plt.colorbar()
#             #     # plt.axis('square')
#             #     # plt.title('correlation map max at ({:.2f},{:.2f})'.format(locx,locy)
#             #     #           + '\nSNR={:.2f}'.format(SNR))
#             #     # plt.show()
                
#             #     # threshold_ratio = 3e-1
#             #     # threshold = numpy.nanmax(corr)*threshold_ratio
#             #     # corr[corr<threshold] = numpy.nan
#             #     # corr -= threshold
                
                
#             #     # yy,xx = numpy.meshgrid(x,x)
                
#             #     # # locx = numpy.nansum(xx*corr*wfs.scaledMask)/numpy.nansum(corr*wfs.scaledMask)
#             #     # # locy = numpy.nansum(yy*corr*wfs.scaledMask)/numpy.nansum(corr*wfs.scaledMask)
#             #     # locx = xx[corr==numpy.nanmax(corr)][0]
#             #     # locy = yy[corr==numpy.nanmax(corr)][0]

#             #     # if (locx == 0) or (locy == 0):
#             #     #     locx = numpy.nan
#             #     #     locy = numpy.nan
                    
#             #     # plt.pcolor(x,x,corr.T)#,vmin=0,vmax=1000)
#             #     # plt.plot(locx,locy,ls='',marker='o',color='r')
#             #     # plt.colorbar()
#             #     # plt.axis('square')
#             #     # plt.title('correlation map max at ({:.2f},{:.2f})'.format(locx,locy))
#             #     # # plt.title('correlation map with {:.2f}% threshold ({:.2f},{:.2f})'.format(threshold_ratio*100,locx,locy))
#             #     # plt.show()
                
#             #     cropped_corr = corr[maxindex[0] - 1 
#             #                         : maxindex[0] + 2,
#             #                         maxindex[1] - 1 
#             #                         : maxindex[1] + 2]
                
#             #     if ((cropped_corr[numpy.isnan(cropped_corr)].sum() <= 0.) 
#             #         and (cropped_corr.size >= 9.)):
                    
#             #         # print(cropped_corr)
#             #         # cropped_x = numpy.arange(-1,2) / wfs.nx_subap_interp
#             #         # cropped_yy,cropped_xx = numpy.meshgrid(cropped_x,cropped_x)
                    
#             #         # initial_guess = numpy.array([0,0,1,1,0])
                    
#             #         # params, cov = curve_fit(parabola2d,
#             #         #                         (cropped_xx.flatten(),
#             #         #                          cropped_yy.flatten()),
#             #         #                         cropped_corr.flatten(),
#             #         #                         initial_guess)
                    
#             #         # true_locx = params[0] + locx
#             #         # true_locy = params[1] + locy
                    
#             #         # M. G. Löfdahl 2010
                    
#             #         a2 = (cropped_corr[1,:].mean() - cropped_corr[-1,:].mean())/2.
#             #         a3 = (cropped_corr[1,:].mean() - 2.*cropped_corr[0,:].mean() + cropped_corr[-1,:].mean())/2.
#             #         a4 = (cropped_corr[:,1].mean() - cropped_corr[:,-1].mean())/2.
#             #         a5 = (cropped_corr[:,1].mean() - 2.*cropped_corr[:,0].mean() + cropped_corr[:,-1].mean())/2.
#             #         a6 = (cropped_corr[1,1] - cropped_corr[-1,1] - cropped_corr[1,-1] + cropped_corr[-1,-1])/4.
                    
#             #         true_locx = (-1 + (2.*a2*a5 - a4*a6)/(a6**2 - 4.*a3*a5)) / wfs.nx_subap_interp  + locx
#             #         true_locy = (-1 + (2.*a3*a4 - a2*a6)/(a6**2 - 4.*a3*a5)) / wfs.nx_subap_interp  + locy

                    
#             #         # plt.pcolor(x,x,corr.T)#,vmin=0,vmax=1000)
#             #         # plt.hlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
#             #         # plt.vlines(numpy.arange(-4,5),-4,4,ls=':',color='r')
#             #         # plt.plot(locx,locy,ls='',marker='o',color='r',label='max pixel')
#             #         # plt.plot(true_locx,true_locy,ls='',marker='o',color='k',label='true max')
#             #         # plt.legend()
#             #         # plt.colorbar()
#             #         # plt.axis('square')
#             #         # plt.title('correlation map max at ({:.2f},{:.2f})'.format(locx,locy)
#             #         #           + '\nSNR={:.2f}'.format(SNR)
#             #         #           #+ '\n x0,y0 fit = ({:.2f},{:.2f})'.format(params[0],params[1])
#             #         #           + '\ntrue max at ({:.2f},{:.2f})'.format(true_locx,true_locy))
#             #         # plt.show()
                    
#             #         if ((numpy.abs(true_locx) < wfs.nx_subaps//2)
#             #             or (numpy.abs(true_locx) < wfs.nx_subaps//2)):
                        
#             #             dm.subapShift[i,wfs_n,0] = true_locx
#             #             dm.subapShift[i,wfs_n,1] = true_locy
                    
#             #     # else:
#             #     #     true_locx = numpy.nan
#             #     #     true_locy = numpy.nan
#             # # else:
#             # #     true_locx = numpy.nan
#             # #     true_locy = numpy.nan
            
            
            
            
#             # #     plt.imshow(wfs.wfsDetectorPlane)
#             # #     plt.title('wfs')
#             # #     plt.colorbar()
#             # #     plt.show()
            
#             # #wfs.slopes = wfs.slopes - zero_iMat[n_wfs_measurments: n_wfs_measurments+wfs.n_measurements]
#             # # make_quiver_plot(wfs.detector_cent_coords,
#             # #                  wfs.slopes
#             # #                  - zero_iMat[n_wfs_measurments
#             # #                              : n_wfs_measurments
#             # #                              + wfs.n_measurements])
                
                
            
#             n_wfs_measurments += wfs.n_measurements

#             # Turn noise back on again if it was turned off
#             if imat_noise is False:
#                 wfs.config.photonNoise = wfs_pnoise
#                 wfs.config.eReadNoise = wfs_rnoise

#         if callback != None:
#             callback()
#     return iMat