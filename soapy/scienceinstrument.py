# Copyright Durham University and Andrew Reeves
# 2014

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
"""
Science Instruments
===================

In this module, several 'science' instruments are defined. These are devices that 
observe a target with the purpose of assessing AO performance
"""

import numpy
import scipy.optimize as opt
import pyfftw

import aotools

from matplotlib import pyplot as plt

from . import logger, lineofsight, numbalib, interp
DTYPE = numpy.float32
CDTYPE = numpy.complex64


class PSFCamera(object):
    """
    A detector observing the Point Spread Function of the telescope

    Parameters:
        soapyConfig (soapy.confParse.Config): Simulation configuration object
        nSci (int, optional): index of this science instrument. default is ``0``
        mask (ndarray, optional): Mask, 1 where telescope is transparent, 0 where opaque. 
    """
    def __init__(self, soapyConfig, nSci=0, mask=None):

        self.soapy_config = soapyConfig
        self.config = self.sciConfig = self.soapy_config.scis[nSci]

        self.simConfig = soapyConfig.sim


        # Get some vital params from config
        self.sim_size = self.soapy_config.sim.simSize
        self.pupil_size = self.soapy_config.sim.pupilSize
        self.sim_pad = self.soapy_config.sim.simPad
        self.fov = self.config.FOV
        self.threads = self.soapy_config.sim.threads
        self.telescope_diameter = self.soapy_config.tel.telDiam
        self.nx_pixels = self.config.pxls

        self.fov_rad = self.config.FOV * numpy.pi / (180. * 3600)

        self.setMask(mask)
        self.pupil_mask = self.mask[self.sim_pad:-self.sim_pad,self.sim_pad:-self.sim_pad]

        self.los = lineofsight.LineOfSight(
                self.config, self.soapy_config,
                propagation_direction=self.config.propagationDir, mask=self.mask) # with mask for sim size
        
        # to reduce conversion, pixel size = simulation/atm pixel size, nx_out_pixels = number of accurate pixels = sim size
        self.los.calcInitParams(
                out_pixel_scale=float(self.telescope_diameter / self.pupil_size),
                nx_out_pixels=self.sim_size
        )
        self.los.allocDataArrays()

        # Init FFT object
        # propagation size for pupil to focal plane of the sci cams
        self.FFTPadding = int(numpy.ceil(
            (self.config.wavelength * self.nx_pixels / self.fov_rad
             / (self.telescope_diameter / self.pupil_size))
            /2)*2)
        
        # N//2 - nout//2
        self.fft_crop_elements = (self.FFTPadding - self.nx_pixels)//2

        # create an FFTW object for fast FFT calculation
        # Must first define input and output arrays, then define the 
        # FFT calculator. This can be multi-threaded and accepts flags
        # Always allow input to be overwritten, option to add other flags from config
        self.fft_input_data = pyfftw.empty_aligned(
                (self.FFTPadding, self.FFTPadding), dtype=CDTYPE)
        self.fft_output_data = pyfftw.empty_aligned(
                (self.FFTPadding, self.FFTPadding), dtype=CDTYPE)
        self.fft_calculator = pyfftw.FFTW(
                self.fft_input_data, self.fft_output_data, axes=(0, 1),
                threads=self.threads,
                flags=(self.config.fftwFlag, "FFTW_DESTROY_INPUT")
        )

        # Allocate some useful arrays
        self.focus_intensity = numpy.zeros((self.nx_pixels, self.nx_pixels), dtype=DTYPE)
        self.detector = numpy.zeros((self.nx_pixels, self.nx_pixels), dtype=DTYPE)
        self.long_exp_image = numpy.zeros_like(self.detector)
        self._long_exp_image = numpy.zeros_like(self.detector)
        self.frame_count = 0

        # Calculate ideal PSF for purposes of strehl calculation
        self.los.frame()
        self.calcFocalPlane()
        # self.bestEField = numpy.copy(self.EField_fov)
        self.psfMax = self.detector.max()
        self.bestPSF = self.detector.copy()/self.psfMax
        
        # plt.imshow(self.bestPSF[self.nx_pixels//2 - 8 : self.nx_pixels//2 + 9,
        #                           self.nx_pixels//2 - 8 : self.nx_pixels//2 + 9],vmin=0,vmax=1)
        # plt.colorbar()
        # plt.show()
        # self.frame_count = 0
        self.longExpStrehl = 0
        self.instStrehl = 0

        # reset to ensure calibration does not affect sim 
        self.reset()


    def reset(self):
        self.longExpStrehl = 0
        self.instStrehl = 0
        self._long_exp_image[:] = 0
        self.long_exp_image[:] = 0
        self.detector[:] = 0
        self.focus_intensity[:] = 0
        self.frame_count = 0


    def setMask(self, mask):
        """
        Sets the pupil mask as seen by the WFS.

        This method can be called during a simulation
        """

        # If supplied use the mask
        if numpy.any(mask):
            self.mask = mask
        else:
            self.mask = aotools.circle(
                    self.pupil_size/2., self.sim_size,
                    )


    def calcFocalPlane(self):
        '''
        Takes the calculated pupil phase, scales for the correct FOV,
        and uses an FFT to transform to the focal plane.
        '''
        
        # If physical propagation, efield should already be the correct
        # size for the Field of View
        self.EField_fov = self.los.EField[
                self.sim_pad: -self.sim_pad,
                self.sim_pad: -self.sim_pad] # crop to pupil size
        
        residual_field = numpy.copy(self.EField_fov)*self.pupil_mask
            
        piston = numpy.nansum(residual_field)/self.pupil_mask
        piston /= numpy.abs(piston)
        
        residual_field /= piston
        residual_field *= self.pupil_mask

        self.residual = residual_field
        
        if self.config.propagationDir == "down":
            self.EField_fov *= self.pupil_mask

        # Get the focal plane using an FFT
        # Reset the FFT from the previous iteration
        self.fft_input_data[:] = 0
        
        # place the array in the centre of the padding
        self.fft_input_data[
                (self.FFTPadding - self.pupil_size)//2:
                (self.FFTPadding + self.pupil_size)//2, 
                (self.FFTPadding - self.pupil_size)//2:
                (self.FFTPadding + self.pupil_size)//2
                ] = self.EField_fov
        # plt.imshow(numpy.abs(self.fft_input_data)**2)
        # plt.title('input data')
        # plt.show()
        # This means we can do a pre-fft shift properly. This is neccessary for anythign that 
        # cares about the EField of the focal plane, not just the intensity pattern
        numbalib.fftshift_2d_inplace(self.fft_input_data)
        self.fft_calculator() # perform FFT
        numbalib.fftshift_2d_inplace(self.fft_output_data)
        
        # plt.imshow(numpy.abs(self.fft_output_data)**2)
        # plt.title('input data')
        # plt.show()
        # where = numpy.where(numpy.abs(self.fft_output_data)**2==(numpy.abs(self.fft_output_data)**2).max())
        # plt.imshow(numpy.abs(self.fft_output_data[where[0][0]-4:where[0][0]+5,where[1][0]-4:where[1][0]+5])**2)
        # plt.title('output data')
        # plt.show()

        if self.fft_crop_elements != 0:
        # Bin down to detector number of pixels
            self.fov_focus_efield = self.fft_output_data[
                    self.fft_crop_elements: -self.fft_crop_elements,
                    self.fft_crop_elements: -self.fft_crop_elements
            ]
        else:
            self.fov_focus_efield = self.fft_output_data


        # Turn complex efield into intensity
        numbalib.abs_squared(self.fov_focus_efield, out=self.focus_intensity)
        self.detector = self.focus_intensity

        # add detector to long exposure image
        self._long_exp_image += self.detector
        self.frame_count += 1

        # Normalise the psf
        # try:
        #     self.detector /= self.psfMax
        # except:
        #     pass
        self.long_exp_image = self._long_exp_image
        
        # plt.imshow(numpy.log10(self.detector/self.detector.max()),vmin=-3,vmax=0)
        # plt.colorbar()
        # plt.title('science detector in log10')
        # plt.show()
        
        # plt.imshow(self.detector/self.detector.max(),vmin=0,vmax=1)
        # plt.colorbar()
        # plt.title('science detector')
        # plt.show()


    def calcInstStrehl(self):
        """
        Calculates the instantaneous Strehl, including TT if configured.
        """
        if self.sciConfig.instStrehlWithTT:
            self.instStrehl = self.detector[self.sciConfig.pxls // 2, self.sciConfig.pxls // 2] / self.psfMax
            self.longExpStrehl = self.long_exp_image[self.sciConfig.pxls //2, self.sciConfig.pxls // 2] / (self.psfMax*self.frame_count)
        else:
            self.instStrehl = self.detector.max() / self.psfMax
            self.longExpStrehl = self.long_exp_image.max() / (self.psfMax*self.frame_count)
            # if self.config.propagationMode == "Physical":
            #     A = self.EField_fov
            #     B = self.bestEField
            #     self.instStrehl = (numpy.abs(numpy.sum(A*numpy.conjugate(B)))**2
            #                        / numpy.abs(numpy.sum(A*numpy.conjugate(A)))
            #                        / numpy.abs(numpy.sum(B*numpy.conjugate(B))))
            #     self.longExpStrehl = self.long_exp_image.max() / self.psfMax
            #     print(self.instStrehl,'not support long strehl for physical yet',self.longExpStrehl)
            # else:
            #     self.instStrehl = self.detector.max() / self.psfMax
            #     self.longExpStrehl = self.long_exp_image.max() / self.psfMax
        
        # P = self.pupil_mask*self.EField_fov
        # Q = self.pupil_mask
        
        # # strehl = (numpy.abs(numpy.sum(P*numpy.conjugate(Q)))**2
        # #                     / numpy.abs(numpy.sum(P*numpy.conjugate(P)))
        # #                     / numpy.abs(numpy.sum(Q*numpy.conjugate(Q))))
        # rytov = numpy.var(numpy.log(numpy.abs(P[numpy.asarray(Q,dtype=bool)])))
        
        # plt.imshow(self.detector[self.nx_pixels//2 - 8 : self.nx_pixels//2 + 9,
        #                          self.nx_pixels//2 - 8 : self.nx_pixels//2 + 9] / self.psfMax,vmin=0,vmax=1)
        # plt.colorbar()
        # plt.title('science detector, Strehl={:.2f}, Rytov={:.2f}'.format(self.instStrehl,rytov))
        # plt.show()


    def calc_wavefronterror(self):
        """
        Calculates the wavefront error across the telescope pupil 
        
        Returns:
             float: RMS WFE across pupil in nm
        """
        if self.config.propagationMode == "Physical":
            
            residual_field = self.residual
            
            # plt.imshow(numpy.angle(residual_field)*self.mask,vmin=-numpy.pi,vmax=numpy.pi)
            # plt.title('science residual rad phys')
            # plt.colorbar()
            # plt.show()
            
            piston = numpy.nansum(residual_field)/self.pupil_mask
            piston /= numpy.abs(piston)
            
            residual_field /= piston
            residual_field *= self.pupil_mask
            
            ms_wfe = numpy.nansum(numpy.square(numpy.angle(residual_field)/self.los.phs2Rad*self.pupil_mask)) / numpy.nansum(self.pupil_mask)
            rms_wfe = numpy.sqrt(ms_wfe)
            # print(rms_wfe)
            
            if self.soapy_config.sim.saveRytov :
            
                P = residual_field
                Q = self.pupil_mask
                
                # strehl = (numpy.abs(numpy.sum(P*numpy.conjugate(Q)))**2
                #                     / numpy.abs(numpy.sum(P*numpy.conjugate(P)))
                #                     / numpy.abs(numpy.sum(Q*numpy.conjugate(Q))))
                rytov = numpy.var(numpy.log(numpy.abs(P[numpy.asarray(Q,dtype=bool)])))
                
                return rms_wfe, rytov
            else:
                return rms_wfe
        
        else:
            res = (self.los.phase.copy() * self.mask) / self.los.phs2Rad
    
            # Piston is mean across aperture
            piston = res.sum() / self.mask.sum()
    
            # remove from WFE measurements as its not a problem
            res -= (piston*self.mask)
    
            ms_wfe = numpy.square(res).sum() / self.mask.sum()
            rms_wfe = numpy.sqrt(ms_wfe)
    
            return rms_wfe


    def frame(self, scrns, correction=None):
        """
        Runs a single science camera frame with one or more phase screens

        Parameters:
            scrns (ndarray, list, dict): One or more 2-d phase screens. Phase in units of nm.
            phaseCorrection (ndarray): Correction phase in nm

        Returns:
            ndarray: Resulting science PSF
        """
        self.los.frame(scrns, correction=correction)
        self.calcFocalPlane()

        self.calcInstStrehl()

        return self.detector


class singleModeFibre(PSFCamera):

    def __init__(self, soapyConfig, nSci=0, mask=None):
        scienceCam.__init__(self, soapyConfig, nSci, mask)

        self.normMask = self.mask / numpy.sqrt(numpy.sum(numpy.abs(self.mask)**2))
        self.fibreSize = opt.minimize_scalar(self.refCouplingLoss, bracket=[1.0, self.sim_size]).x
        self.refStrehl = 1.0 - self.refCouplingLoss(self.fibreSize)
        self.fibre_efield = self.fibreEfield(self.fibreSize)
        print("Coupling efficiency: {0:.3f}".format(self.refStrehl))


    def fibreEfield(self, size):
        fibre_efield = aotools.gaussian2d((self.sim_size, self.sim_size), (size, size))
        fibre_efield /= numpy.sqrt(numpy.sum(numpy.abs(aotools.gaussian2d((self.sim_size*3, self.sim_size*3), (size, size)))**2))
        return fibre_efield


    def refCouplingLoss(self, size):
        return 1.0 - numpy.abs(numpy.sum(self.fibreEfield(size) * self.normMask))**2


    def calcInstStrehl(self):
        self.instStrehl = numpy.abs(numpy.sum(self.fibre_efield * self.los.EField * self.normMask))**2


# Compatability with older versions
scienceCam = ScienceCam = PSF = PSFCamera
