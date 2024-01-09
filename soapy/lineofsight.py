"""
A generalised module to provide phase or the EField through a "Line Of Sight"

Line of Sight Object
====================
The module contains a 'lineOfSight' object, which calculates the resulting phase or complex amplitude from propogating through the atmosphere in a given
direction. This can be done using either geometric propagation, where phase is simply summed for each layer, or physical propagation, where the phase is propagated between layers using an angular spectrum propagation method. Light can propogate either up or down.

The Object takes a 'config' as an argument, which is likely to be the same config object as the module using it (WFSs, ScienceCams, or LGSs). It should contain paramters required, such as the observation direction and light wavelength. The `config` also determines whether to use physical or geometric propagation through the 'propagationMode' parameter.

Examples::

    from soapy import confParse, lineofsight

    # Initialise a soapy conifuration file
    config = confParse.loadSoapyConfig('conf/sh_8x8.py')

    # Can make a 'LineOfSight' for WFSs
    los = lineofsight.LineOfSight(config.wfss[0], config)

    # Get resulting complex amplitude through line of sight
    EField = los.frame(some_phase_screens)

"""

import numpy

import aotools
from aotools import opticalpropagation
import copy 

from . import logger, interp
from . import numbalib

from matplotlib import pyplot as plt
import numpy as np

DTYPE = numpy.float32
CDTYPE = numpy.complex64

# Python3 compatability
try:
    xrange
except NameError:
    xrange = range

RAD2ASEC = 206264.849159
ASEC2RAD = 1./RAD2ASEC

class LineOfSight(object):
    """
    A "Line of sight" through a number of turbulence layers in the atmosphere, observing ing a given direction.

    Parameters:
        config: The soapy config for the line of sight
        simConfig: The soapy simulation config object
        propagation_direction (str, optional): Direction of light propagation, either `"up"` or `"down"`
        outPxlScale (float, optional): The EField pixel scale required at the output (m/pxl)
        nOutPxls (int, optional): Number of pixels to return in EFIeld
        mask (ndarray, optional): Mask to apply at the *beginning* of propagation
        metaPupilPos (list, dict, optional): A list or dictionary of the meta pupil position at each turbulence layer height ub metres. If None, works it out from GS position.
    """
    def __init__(
            self, config, soapyConfig,
            propagation_direction="down", out_pixel_scale=None,
            nx_out_pixels=None, mask=None, metaPupilPos=None):

        self.config = config
        self.soapy_config = soapyConfig
        self.pupil_size = self.soapy_config.sim.pupilSize
        self.phase_pixel_scale = 1./self.soapy_config.sim.pxlScale
        self.sim_size = self.soapy_config.sim.simSize
        self.wavelength = self.config.wavelength
        self.telescope_diameter = self.soapy_config.tel.telDiam
        self.propagation_direction = propagation_direction

        self.source_altitude = self.height

        self.nx_scrn_size = self.soapy_config.sim.scrnSize
        self.n_layers = self.soapy_config.atmos.scrnNo
        self.layer_altitudes = self.soapy_config.atmos.scrnHeights

        self.n_dm = self.soapy_config.sim.nDM
        self.dm_altitudes = numpy.array([self.soapy_config.dms[dm_n].altitude for dm_n in range(self.n_dm)])

        # If the line of sight has a launch position, must include in calculations! Conver to metres from centre of telescope
        try:
            self.launch_position = numpy.array(self.config.launchPosition) * self.telescope_diameter/2.
        except AttributeError:
            self.launch_position = numpy.array([0, 0])

        self.simConfig = soapyConfig.sim
        self.atmosConfig = soapyConfig.atmos

        self.mask = mask

        self.calcInitParams(out_pixel_scale, nx_out_pixels)

        self.allocDataArrays()

        # Can be set to use other values as metapupil position
        self.metaPupilPos = metaPupilPos


    # Some attributes for compatability between WFS and others
    @property
    def height(self):
        try:
            return self.config.height
        except AttributeError:
            return self.config.GSHeight

    @height.setter
    def height(self, height):
        try:
            self.config.height
            self.config.height = height
        except AttributeError:
            self.config.GSHeight
            self.config.GSHeight = height

    @property
    def position(self):
        try:
            return self.config.position
        except AttributeError:
            return self.config.GSPosition

    @position.setter
    def position(self, position):
        try:
            self.config.position
            self.config.position = position
        except AttributeError:
            self.config.GSPosition
            self.config.GSPosition = position

############################################################
# Initialisation routines


    def calcInitParams(self, out_pixel_scale=None, nx_out_pixels=None):
        """
        Calculates some parameters required later

        Parameters:
            outPxlScale (float): Pixel scale of required phase/EField (metres/pxl)
            nOutPxls (int): Size of output array in pixels
        """
        logger.debug("Calculate LOS Init PArams!")
        # Convert phase deviation to radians at wfs wavelength.
        # (currently in nm remember...?)
        self.phs2Rad = 2*numpy.pi/(self.wavelength * 10**9)

        # Get the size of the phase required by the system
        self.in_pixel_scale = self.phase_pixel_scale

        if out_pixel_scale is None:
            self.out_pixel_scale = self.phase_pixel_scale
        else:
            self.out_pixel_scale = out_pixel_scale
            
        # try:
        #     self.fov = self.config.subapFOV * 2*np.pi/360./3600.
        # except:
        #     self.fov = self.config.FOV * 2*np.pi/360./3600.
        self.fov = 0
        
        
            
        if nx_out_pixels is None:
            self.nx_out_pixels = self.simConfig.simSize
        else:
            self.nx_out_pixels = nx_out_pixels
            self.max_input_pixel = nx_out_pixels
            
        if self.config.propagationMode == 'Physical':
            
            self.max_grid_diffraction_angle = self.wavelength/2./self.out_pixel_scale
            # self.max_diffraction_angle = self.soapy_config.sim.max_diffraction_angle + self.fov/2.
            # self.max_simulation_angle = None
            self.max_simulation_angle = self.max_grid_diffraction_angle
            

            
            # ***** !!!!!
            # assume same pixel_scale throughout everything!!!!
            # this is a big threat to soapy as soapy is so flexible
            # things don't always share the same pixel scale
            
            self.max_input_pixel = int(numpy.ceil(
                (self.nx_out_pixels
                 + numpy.max(numpy.concatenate([self.dm_altitudes,self.layer_altitudes,[0]]))
                 * (2*self.max_grid_diffraction_angle)/self.out_pixel_scale)
                /2)*2)
            
            
            
            self.nx_prop_pixels = int(round(
                numpy.max([
                    
                    2*self.max_input_pixel,
                    
                    self.wavelength*numpy.max(numpy.concatenate([
                        self.layer_altitudes,self.dm_altitudes,[0]]))/self.out_pixel_scale**2
                    
                    ])
                /2)*2)
            
            # the next even number of pixels that contains
            # pupil, ( fov + 2*diffraction_per_side )scaled to max height
            # and the largest angular-spectrum in OTF = wvl*z/delta**2
            
            
            
            self.low_buf = (self.nx_prop_pixels - self.nx_out_pixels)//2
            self.high_buf = (self.nx_prop_pixels + self.nx_out_pixels)//2
            
            self.pad = (self.nx_prop_pixels - self.max_input_pixel)//2
            
            test_result = test_propagation_parameters(self.wavelength,self.layer_altitudes,self.dm_altitudes,
                                        self.nx_prop_pixels,
                                        self.max_input_pixel,
                                        self.nx_out_pixels*self.pupil_size/self.sim_size,
                                        self.out_pixel_scale,
                                        self.max_simulation_angle)
            
            if (test_result == False):
                print('Propagation Parameter need adjustment')
            # else:
            #     print('Pass propagation criteria list')
            
            
        else:
            # if not physical, all the new parameters should equal to the only old parameters   
            self.nx_prop_pixels = self.nx_out_pixels
            
        temp2 = int(numpy.floor(self.nx_out_pixels*self.pupil_size/self.sim_size))
        temp1 = temp2//2
            
        self.plot_mask = aotools.circle(temp1,temp2)
            
        if False:#self.mask is not None:
            self.outMask = interp.zoom_rbs (
                    self.mask, self.nx_out_pixels, order=1)#.round()
        else:
            self.outMask = None
            # self.outMask = interp.zoom(
            #         self.mask, self.nx_out_pixels)#.round()

        self.output_phase_diameter = self.nx_out_pixels * self.out_pixel_scale
        

        # Calculate coords of phase at each altitude
        self.layer_metapupil_coords = numpy.zeros((self.n_layers, 2, self.max_input_pixel))
        for i in range(self.n_layers):
            x1, x2, y1, y2 = self.calculate_altitude_coords(self.layer_altitudes[i])
            self.layer_metapupil_coords[i, 0] = numpy.linspace(x1, x2-1, self.max_input_pixel)
            self.layer_metapupil_coords[i, 1] = numpy.linspace(y1, y2-1, self.max_input_pixel)

        # ensure coordinates aren't out of bounds for interpolation
        self.layer_metapupil_coords = self.layer_metapupil_coords.clip(0, self.nx_scrn_size - 1.000000001)

        # Calculate coords of phase at each DM altitude
        self.dm_metapupil_coords = numpy.zeros((self.n_dm, 2, self.max_input_pixel))
        for i in range(self.n_dm):
            x1, x2, y1, y2 = self.calculate_altitude_coords(self.dm_altitudes[i])
            self.dm_metapupil_coords[i, 0] = numpy.linspace(x1, x2-1, self.max_input_pixel)
            self.dm_metapupil_coords[i, 1] = numpy.linspace(y1, y2-1, self.max_input_pixel)
        self.dm_metapupil_coords = self.dm_metapupil_coords.clip(0, self.nx_scrn_size - 1.000000001)

        self.radii = None
        
        # Relavent phase  centred on the line of sight direction
        self.phase_screens = numpy.zeros((self.n_layers, self.max_input_pixel, self.max_input_pixel))
        # Buffer for corection across fulll FOV
        self.correction_screens = numpy.zeros((self.n_dm, self.max_input_pixel, self.max_input_pixel))
        # Relavent correction centred on the line of sight direction
        self.phase_correction = numpy.zeros((self.max_input_pixel, self.max_input_pixel))
        if self.config.propagationMode == 'Physical':
            self.phase_screens_buf = numpy.zeros((self.n_layers, self.nx_prop_pixels, self.nx_prop_pixels))
            self.correction_screens_buf = numpy.zeros((self.n_dm, self.nx_prop_pixels, self.nx_prop_pixels))
            self.phase_correction_buf = numpy.zeros((self.nx_prop_pixels, self.nx_prop_pixels))

        self.allocDataArrays()

    def calculate_altitude_coords(self, layer_altitude):
        """
        Calculate the co-ordinates of vertices of fo the meta-pupil at altitude given a guide star
        direction and source altitude

        Paramters:
            layer_altitude (float): Altitude of phase layer
        """
        direction_radians = ASEC2RAD * numpy.array(self.position)

        centre = (direction_radians * layer_altitude)

        # If propagating up must account for launch position
        if self.propagation_direction == "up":
            if self.source_altitude == 0:
                centre += self.launch_position
            else:
                centre += self.launch_position * (1 - layer_altitude/self.source_altitude)
                
        if self.config.propagationMode == 'Physical':
            if self.source_altitude != 0:
                meta_pupil_size = (self.max_input_pixel*self.out_pixel_scale * (1 - layer_altitude / self.source_altitude))
                # print('currently not supporting this')
            else:
                meta_pupil_size = self.max_input_pixel*self.out_pixel_scale
        else:
            if self.source_altitude != 0:
                meta_pupil_size = self.output_phase_diameter * (1 - layer_altitude / self.source_altitude)
            else:
                meta_pupil_size = self.output_phase_diameter

        x1 = ((centre[0] - meta_pupil_size / 2.) / self.in_pixel_scale) + self.nx_scrn_size / 2.
        x2 = ((centre[0] + meta_pupil_size / 2.) / self.in_pixel_scale) + self.nx_scrn_size / 2.
        y1 = ((centre[1] - meta_pupil_size / 2.) / self.in_pixel_scale) + self.nx_scrn_size / 2.
        y2 = ((centre[1] + meta_pupil_size / 2.) / self.in_pixel_scale) + self.nx_scrn_size / 2.

        logger.debug("Layer Altitude: {}".format(layer_altitude))
        logger.debug("Coords: x1: {}, x2: {}, y1: {},  y2: {}".format(x1, x2, y1, y2))

        return x1, x2, y1, y2

    def allocDataArrays(self):
        """
        Allocate the data arrays the LOS will require

        Determines and allocates the various arrays the LOS will require to
        avoid having to re-alloc memory during the running of the LOS and
        keep it fast. This includes arrays for phase
        and the E-Field across the LOS
        """
        self.phase = numpy.zeros([self.nx_out_pixels] * 2, dtype=DTYPE)
        self.EField = numpy.ones([self.nx_out_pixels] * 2, dtype=CDTYPE)
        self.residual = numpy.zeros([self.nx_out_pixels] * 2, dtype=DTYPE)
        self.correction_EField = numpy.copy(self.EField)
        self.residual_EField = numpy.copy(self.EField)
        
        if self.config.propagationMode == 'Physical':
            self.EField_buf = numpy.ones([self.nx_prop_pixels] * 2, dtype=CDTYPE)

######################################################

    def zeroData(self, **kwargs):
        """
        Sets the phase and complex amp data to zero
        """
        self.EField[:] = 1
        self.phase[:] = 0
        self.phase_screens[:] = 0
        self.correction_screens[:] = 0
        self.correction_EField[:] = 1
        
        if self.config.propagationMode == 'Physical':
            self.EField_buf[:] = 1
            self.phase_screens_buf[:] = 0
            self.correction_screens_buf[:] = 0
        
    def makePhase(self, radii=None, apos=None):
        """
        Generates the required phase or EField. Uses difference approach depending on whether propagation is geometric or physical
        (makePhaseGeometric or makePhasePhys respectively)

        Parameters:
            radii (dict, optional): Radii of each meta pupil of each screen height in pixels. If not given uses pupil radius.
            apos (ndarray, optional):  The angular position of the GS in radians. If not set, will use the config position
        """
        
        # Check if geometric or physical
        if self.config.propagationMode == "Physical":
            return self.makePhasePhys(radii)
        else:
            return self.makePhaseGeometric(radii)


    def makePhaseGeometric(self, radii=None, apos=None):
        '''
        Creates the total phase along line of sight offset by a given angle using a geometric ray tracing approach

        Parameters:
            radii (dict, optional): Radii of each meta pupil of each screen height in pixels. If not given uses pupil radius.
            apos (ndarray, optional):  The angular position of the GS in radians. If not set, will use the config position
        '''

        for i in range(self.scrns.shape[0]):
            numbalib.bilinear_interp(
                self.scrns[i], self.layer_metapupil_coords[i, 0], self.layer_metapupil_coords[i, 1],
                self.phase_screens[i], bounds_check=False)
            
        self.atmos_phase = self.phase_screens.sum(0)

        # Convert phase to radians
        self.atmos_phase *= self.phs2Rad

        # Change sign if propagating up
        # if self.propagation_direction == 'up':
        #     self.atmos_phase *= -1

        self.phase[:] += self.atmos_phase
        self.EField[:] *= numpy.exp(1j*self.atmos_phase)
        
        if self.outMask is not None:
            self.EField[:] = np.copy(self.EField[:] * self.outMask)

        return self.EField


    def makePhasePhys(self, radii=None, apos=None):
        '''
        Finds total line of sight complex amplitude by propagating light through phase screens

        Parameters:
            radii (dict, optional): Radii of each meta pupil of each screen height in pixels. If not given uses pupil radius.
            apos (ndarray, optional):  The angular position of the GS in radians. If not set, will use the config position
        '''
        
        for i in range(self.scrns.shape[0]):
            numbalib.bilinear_interp(
                self.scrns[i], self.layer_metapupil_coords[i, 0], self.layer_metapupil_coords[i, 1],
                self.phase_screens[i], bounds_check=False)
            self.phase_screens_buf[i] = numpy.pad(self.phase_screens[i],
                                                  (self.pad,self.pad),
                                                  mode='constant',
                                                  constant_values=(0,0))
            
            # if (self.scrns.sum() != 0):
            #     if (self.scrns[i].sum() != 0):
            #         plt.imshow(self.scrns[i])
            #         plt.colorbar()
            #         plt.title('original full screen {:}'.format(i)
            #                   + ' : {}'.format(self.config.type))
            #         plt.show()
            #         plt.imshow(self.phase_screens[i])
            #         plt.colorbar()
            #         plt.title('original los phase screen {:}'.format(i)
            #                   + ' : {}'.format(self.config.type))
            #         plt.show()
            #         plt.imshow(self.phase_screens_buf[i])
            #         plt.colorbar()
            #         plt.title('original buffered los phase screen {:}'.format(i)
            #                   + ' : {}'.format(self.config.type))
            #         plt.show()
            
        self.EField_buf[:] = physical_atmosphere_propagation(
            self.phase_screens_buf, self.outMask, self.layer_altitudes, self.source_altitude,
            self.wavelength, self.out_pixel_scale,
            propagation_direction=self.propagation_direction,
            input_efield=self.EField_buf, max_angle=None)
        
        # self.phase[:] = numpy.angle(self.EField[:])
        
        
        self.EField = self.EField_buf[self.low_buf:self.high_buf,
                                      self.low_buf:self.high_buf]
        
        if self.outMask is not None:
            self.EField[:] = np.copy(self.EField[:] * self.outMask)

        return self.EField#,self.EField_buf 


    def performCorrection(self, correction):
        """
        Corrects the aberrated line of sight with some given correction phase
        
        Parameters:
            correction (list or ndarray): either 2-d array describing correction, or list of correction arrays
        """
        
        if self.config.propagationMode == 'Physical':
            self.performCorrectionPhysical(correction)
        else:
            self.performCorrectionGeometric(correction)
        
    def performCorrectionGeometric(self, correction):
        
        for i in range(correction.shape[0]):
            numbalib.bilinear_interp(
                correction[i], self.dm_metapupil_coords[i, 0], self.dm_metapupil_coords[i, 1],
                self.correction_screens[i], bounds_check=False)
            
        # original_state = numpy.copy(self.EField)
        
        self.correction_screens.sum(0, out=self.phase_correction)

        self.phase_correction *= self.phs2Rad
        # Correct EField
        self.EField *= numpy.exp(-1j * self.phase_correction)

        if self.outMask is not None:
            mean_phase = np.sum(self.EField*self.outMask)/np.sum(self.outMask)
        else:
            mean_phase = np.mean(self.EField)
        mean_phase /= np.abs(mean_phase)
        # mean_phase = 1
        
        self.EField /= mean_phase

        # Also correct phase in case its required
        
        self.residual = (self.phase - self.phase_correction - np.angle(mean_phase)) / self.phs2Rad

        self.phase = self.residual * self.phs2Rad
        
        # plot = False
        # if ((correction is not None)
        #     and (self.scrns.sum() != 0)
        #     and (self.config.type == 'ShackHartmann')
        #     and plot):
            
        #     A = original_state.shape[-1]
        #     B = self.nx_out_pixels
            
        #     prop_atm_phase = self.plot_mask*np.angle(original_state[(A-B)//2:(A+B)//2,
        #                                         (A-B)//2:(A+B)//2]/mean_phase)
        #     plt.imshow(prop_atm_phase, vmin=-numpy.pi,vmax=numpy.pi)
        #     plt.title('atm phase geometric')
        #     plt.colorbar()
        #     plt.show()
        #     prop_atm_intensity = self.plot_mask * np.abs(original_state[(A-B)//2:(A+B)//2,
        #                                         (A-B)//2:(A+B)//2]/mean_phase)**2
        #     plt.imshow(prop_atm_intensity,
        #                 vmin=0,vmax=3)
        #     plt.title('atm intensity geometric')
        #     plt.colorbar()
        #     plt.show()
            
        #     prop_dm_phase = self.plot_mask*np.angle(numpy.exp(1j*self.phase_correction[(A-B)//2:(A+B)//2,
        #                                         (A-B)//2:(A+B)//2])
        #                         /mean_phase)
        #     plt.imshow(prop_dm_phase,vmin=-numpy.pi,vmax=numpy.pi)
        #     plt.title('dm phase geometric')
        #     plt.colorbar()
        #     plt.show()
            
        #     prop_dm_intensity = self.plot_mask * np.abs(numpy.exp(1j*self.phase_correction[(A-B)//2:(A+B)//2,
        #                                         (A-B)//2:(A+B)//2])
        #                       /mean_phase)**2
        #     plt.imshow(prop_dm_intensity,vmin=0,vmax=3)
        #     plt.title('dm intensity geometric')
        #     plt.colorbar()
        #     plt.show()
            
        #     # not true as this still runs with recon from wfs/dm with propagation
        #     # regular_res_phase = self.plot_mask*(np.angle(np.exp(1j*(
        #     #     regular_atm - regular_dm))/mean_phase))
        #     # plt.imshow(regular_res_phase,vmin=-numpy.pi,vmax=numpy.pi)
        #     # plt.title('residal phase no prop')
        #     # plt.colorbar()
        #     # plt.show()
            
        #     prop_res_phase = self.plot_mask*np.angle(self.EField[(A-B)//2:(A+B)//2,
        #                                         (A-B)//2:(A+B)//2])
        #     plt.imshow(prop_res_phase,vmin=-numpy.pi,vmax=numpy.pi)
        #     plt.title('res phase geometric')
        #     plt.colorbar()
        #     plt.show()
            
        #     prop_res_phase = self.plot_mask*np.abs(self.EField[(A-B)//2:(A+B)//2,
        #                                         (A-B)//2:(A+B)//2])**2
        #     plt.imshow(prop_res_phase,vmin=0,vmax=3)
        #     plt.title('res intensity geometric')
        #     plt.colorbar()
        #     plt.show()
            
    
    def performCorrectionPhysical(self, correction):
        # either up or down, still need to start from pupil, ... right?
        # anyway for 'down' definitely need to pick up from the pupil
        for i in range(correction.shape[0]):
            numbalib.bilinear_interp(
                correction[i], self.dm_metapupil_coords[i, 0], self.dm_metapupil_coords[i, 1],
                self.correction_screens[i], bounds_check=False)
            self.correction_screens_buf[i] = numpy.pad(self.correction_screens[i],
                                                       (self.pad,self.pad),
                                                       mode='constant',
                                                       constant_values=(0,0))
            # if (correction[i].sum() != 0):
            #     plt.imshow(correction[i])
            #     plt.colorbar()
            #     plt.title('original full dm {:}'.format(i)
            #               + ' : {}'.format(self.config.type))
            #     plt.show()
            #     plt.imshow(self.correction_screens[i])
            #     plt.colorbar()
            #     plt.title('original los dm {:}'.format(i)
            #               + ' : {}'.format(self.config.type))
            #     plt.show()
            #     plt.imshow(self.correction_screens_buf[i])
            #     plt.colorbar()
            #     plt.title('original buffered dm {:}'.format(i)
            #               + ' : {}'.format(self.config.type))
            #     plt.show()
        
        original_state = numpy.copy(self.EField_buf[self.low_buf:self.high_buf,
                                            self.low_buf:self.high_buf])
        
        physical_correction_propagation(
            self.correction_screens_buf, None, self.dm_altitudes,
            self.wavelength, self.out_pixel_scale,
            input_efield=self.EField_buf, max_angle=None)
        self.EField = numpy.copy(self.EField_buf[self.low_buf:self.high_buf,
                                      self.low_buf:self.high_buf])
        
        self.residual_EField = numpy.copy(self.EField)
        self.correction_EField = numpy.copy(original_state/self.EField)
        
        if self.outMask is not None:
            mean_phase = np.sum(self.EField*self.outMask)/np.sum(self.outMask)
        else:
            mean_phase = np.mean(self.EField)
        mean_phase /= np.abs(mean_phase)
        # mean_phase = 1
        
        self.EField /= mean_phase
        
        self.residual = np.angle(self.EField) / self.phs2Rad
        self.phase = self.residual * self.phs2Rad
        
        plot = False#((self.scrns.sum() != 0) or (correction.sum() != 0))
        if plot:
            A = original_state.shape[-1]
            B = self.plot_mask.shape[0]
            
            if (self.scrns.sum() != 0):
                
                regular_atm = self.plot_mask*(np.angle(np.exp(1j*(self.phase_screens_buf[:,
                                                                  self.low_buf:self.high_buf,
                                                                  self.low_buf:self.high_buf][
                                                                      :,(A-B)//2:(A+B)//2,
                                                                      (A-B)//2:(A+B)//2]).sum(0))/mean_phase))
                plt.imshow(regular_atm)#, vmin=-numpy.pi,vmax=numpy.pi)
                plt.title('atm phase no prop : {}'.format(self.config.type))
                plt.colorbar()
                plt.show()
                
                
                if (self.layer_altitudes != 0).any():
                
                    prop_atm_phase = self.plot_mask*np.angle(original_state[(A-B)//2:(A+B)//2,
                                                        (A-B)//2:(A+B)//2]/mean_phase)
                    plt.imshow(prop_atm_phase, vmin=-numpy.pi,vmax=numpy.pi)
                    plt.title('atm phase : {}'.format(self.config.type))
                    plt.colorbar()
                    plt.show()
                    # prop_atm_intensity = self.plot_mask * np.abs(original_state[(A-B)//2:(A+B)//2,
                    #                                     (A-B)//2:(A+B)//2]/mean_phase)**2
                    # plt.imshow(prop_atm_intensity,
                    #             vmin=0,vmax=3)
                    # plt.title('atm intensity')
                    # plt.colorbar()
                    # plt.show()
            
                if (correction is not None):
                    
                    regular_dm = self.plot_mask*np.angle(np.exp(1j*(self.correction_screens_buf[:,self.low_buf:self.high_buf,
                                                  self.low_buf:self.high_buf])[:,(A-B)//2:(A+B)//2,
                                                        (A-B)//2:(A+B)//2].sum(0))/mean_phase)
                    plt.imshow(regular_dm,vmin=-numpy.pi,vmax=numpy.pi)
                    # plt.title('dm phase no prop')
                    plt.title('dm phase at altitude : {}'.format(self.config.type))
                    plt.colorbar()
                    plt.show()
                    
                    # if True:#(self.dm_altitudes != 0).any():
                    
                    prop_dm_phase = self.plot_mask*np.angle(self.correction_EField[(A-B)//2:(A+B)//2,
                                                        (A-B)//2:(A+B)//2]
                                        /mean_phase)
                    plt.imshow(prop_dm_phase,vmin=-numpy.pi,vmax=numpy.pi)
                    plt.title('dm phase : {}'.format(self.config.type))
                    plt.colorbar()
                    plt.show()
                    
                    prop_dm_intensity = self.plot_mask * np.abs(self.correction_EField[(A-B)//2:(A+B)//2,
                                                        (A-B)//2:(A+B)//2]
                                      /mean_phase)**2
                    plt.imshow(prop_dm_intensity,vmin=0,vmax=3)
                    plt.title('dm intensity : {}'.format(self.config.type))
                    plt.colorbar()
                    plt.show()
                    
                # if ((correction is not None) or (self.scrns.sum() != 0)):
                    
                    # not true as this still runs with recon from wfs/dm with propagation
                    # regular_res_phase = self.plot_mask*(np.angle(np.exp(1j*(
                    #     regular_atm - regular_dm))/mean_phase))
                    # plt.imshow(regular_res_phase,vmin=-numpy.pi,vmax=numpy.pi)
                    # plt.title('residal phase no prop')
                    # plt.colorbar()
                    # plt.show()
                    
                    P = self.plot_mask*self.EField[(A-B)//2:(A+B)//2,
                                                   (A-B)//2:(A+B)//2]
                    Q = self.plot_mask
                    
                    strehl = (numpy.abs(numpy.sum(P*numpy.conjugate(Q)))**2
                                       / numpy.abs(numpy.sum(P*numpy.conjugate(P)))
                                       / numpy.abs(numpy.sum(Q*numpy.conjugate(Q))))
                    
                    prop_res_phase = self.plot_mask*np.angle(self.EField[(A-B)//2:(A+B)//2,
                                                        (A-B)//2:(A+B)//2])
                    plt.imshow(prop_res_phase,vmin=-numpy.pi,vmax=numpy.pi)
                    plt.title('res phase,\n'
                              + '{}: strehl={}'.format(self.config.type, strehl))
                    plt.colorbar()
                    plt.show()
                    
                    if (self.layer_altitudes != 0).any() or (self.dm_altitudes != 0).any():
                    
                        prop_res_intensity = self.plot_mask*np.abs(self.EField[(A-B)//2:(A+B)//2,
                                                            (A-B)//2:(A+B)//2])**2
                        plt.imshow(prop_res_intensity,vmin=0,vmax=3)
                        plt.colorbar()
                        plt.title('res intensity,\n'
                                  + '{}: strehl={}'.format(self.config.type, strehl))
                        plt.show()
            

    def frame(self, scrns=None, correction=None):
        '''
        Runs one frame through a line of sight

        Finds the phase or complex amplitude through line of sight for a
        single simulation frame, with a given set of phase screens and
        some optional correction. 
        If scrns is ``None``, then light is propagated with no phase.

        Parameters:
            scrns (list): A list or dict containing the phase screens
            correction (ndarray, optional): The correction term to take from the phase screens before the WFS is run.
            read (bool, optional): Should the WFS be read out? if False, then WFS image is calculated but slopes not calculated. defaults to True.

        Returns:
            ndarray: WFS Measurements
        '''

        self.zeroData()

        # If we propagate up, must do correction first!
        if (self.propagation_direction == "up") and (correction is not None):
            self.performCorrection(correction)

        # Now do propagation through atmospheric turbulence
        if scrns is not None:
            if scrns.ndim==2:
                scrns.shape = 1, scrns.shape[0], scrns.shape[1]
            self.scrns = scrns
        else: # If no scrns, just assume no turbulence
            self.scrns = numpy.zeros(
                    (self.n_layers, self.nx_scrn_size, self.nx_scrn_size))

        self.makePhase(self.radii)
        self.residual = self.phase
        # If propagating down, do correction last
        if (self.propagation_direction == "down") and (correction is not None):
            self.performCorrection(correction)
        
        # print(np.angle(self.EField.mean()))
        
        return self.residual


# proper propagation without limiting angular-spectrum
def test_propagation_parameters(wvl,turbHs,DMHs,
                            N,n_in,n_out,scale,
                            angle):
    n_conditions = 6
    heights = numpy.unique(numpy.abs(numpy.concatenate((turbHs,DMHs,[0]))))
    MAXZ = heights.max() - heights.min()
    conditions = numpy.zeros((n_conditions), dtype=bool)
    MAXANGLE = wvl/2./scale
    if angle is None:
        angle = MAXANGLE
    
    # enough coverage of input for output
    INPUTSIZE = n_in*scale
    SPREAD = 2.*angle*MAXZ
    OUTPUTSIZE = n_out*scale
    conditions[0] = ((INPUTSIZE >= OUTPUTSIZE + SPREAD)
                     or numpy.isclose(INPUTSIZE, OUTPUTSIZE + SPREAD, rtol=1e-2))
    if conditions[0] == False:
        print('Currently having {}px of input, and need total {}px output,'.format(INPUTSIZE/scale,OUTPUTSIZE/scale+SPREAD/scale)
              + '\nwhere {}px for accurate output and {}px of buffer,'.format(OUTPUTSIZE/scale,SPREAD/scale))
    
    # enough screen size to accommodate all data
    SCREENSIZE = N*scale
    conditions[1] = ((SCREENSIZE >= INPUTSIZE)
                     or numpy.isclose(SCREENSIZE,INPUTSIZE,rtol=1e-2))
    if conditions[1] == False:
        print('Currently having {}m of screen, while it needs {}m.'.format(
            SCREENSIZE,INPUTSIZE))
    
    # no wrapping on output
    DATASPACE = (n_in/2. + n_out/2.)*scale
    WRAPPING = angle*MAXZ
    AVAILABLESPACES = N*scale
    USE = DATASPACE + WRAPPING
    conditions[2] = ((AVAILABLESPACES >= USE)
                     or numpy.isclose(AVAILABLESPACES,USE,rtol=1e-2))
    if conditions[2] == False:
        print('Currently there is a wrapping effect in the desired accurate region of the output'
              +'\nThere is {}m avaliable spaces, while it needs {}m'.format(
                  AVAILABLESPACES,USE))
        
    # no false edge reaching pupil (inward direction while previous condition is outward)
    BUFFER = (n_in - n_out)/2*scale
    WRAPPING = angle*MAXZ
    conditions[3] = ((BUFFER >= WRAPPING) or numpy.isclose(BUFFER,WRAPPING,rtol=1e-2))
    if conditions[3] == False:
        print('There is an inward wrapping. Need more input.')
        print('suggest input of {}px. currently have {}px'.format(n_out + 2*MAXANGLE*MAXZ/scale,n_in))
    
    # enough coverage in angular-spectrum
    conditions[4] = ((MAXANGLE >= angle) or numpy.isclose(MAXANGLE,angle,rtol=1e-2))
    if conditions[4] == False:
        print('Cannot simulate the physically max angular spectrum'
              + '\nSimulation Max is {}arcsec, while {}arcsec is needed'.format(
                  MAXANGLE*206265,angle*206265))
    
    # no aliasing in optical transfer function
    REQUIREPIXEL = wvl*MAXZ/scale**2
    conditions[5] = ((N >= REQUIREPIXEL) or numpy.isclose(N,REQUIREPIXEL,rtol=1e-2))
    if conditions[5] == False:
        print('There is aliasing in optical transfer function'
              + '\nRequire simulating {} pixels, currently at {} pixels'.format(
                  REQUIREPIXEL,N))
    
    condition = conditions.all()
    
    return condition

def physical_atmosphere_propagation(
            phase_screens, output_mask, layer_altitudes, source_altitude,
            wavelength, output_pixel_scale,
            propagation_direction="up", input_efield=None, max_angle=None):
    '''
    Finds total line of sight complex amplitude by propagating light through phase screens

    If the source altitude is infinity (denoted as 0), then the result of the propagation is
    the

    Parameters:
        
    '''

    scrnNo = len(phase_screens)
    z_total = 0
    scrnRange = range(0, scrnNo)

    nx_output_pixels = phase_screens[0].shape[0]

    phs2Rad = 2 * numpy.pi / (wavelength * 10 ** 9)

    EFieldBuf = input_efield
    if input_efield is None:
        EFieldBuf = numpy.exp(
                1j*numpy.zeros((nx_output_pixels,) * 2)).astype(CDTYPE)

    # Get initial up/down dependent params
    if propagation_direction == "up":
        ht = 0
        ht_final = source_altitude
        # if ht_final==0:
        #     raise ValueError("Can't propagate up to infinity")
        scrnAlts = layer_altitudes
        # If propagating up from telescope, apply mask to the EField
        EFieldBuf *= output_mask
        logger.debug("Create EField Buf of mask")

    else:
        ht = layer_altitudes[scrnNo-1]
        ht_final = 0
        scrnAlts = layer_altitudes[::-1]
        phase_screens = phase_screens[::-1]
        logger.debug("Create EField Buf of zero phase")

    # Propagate to first phase screen (if not already there)
    if ht!=scrnAlts[0]:
        logger.debug("propagate to first phase screen")
        z = abs(scrnAlts[0] - ht)
        z_total += z
        if max_angle is None:
            EFieldBuf[:] = opticalpropagation.angularSpectrum(
                        EFieldBuf, wavelength,
                        output_pixel_scale, output_pixel_scale, z)
        else:
            EFieldBuf[:] = opticalpropagation.limited_angularSpectrum(
                        EFieldBuf, wavelength,
                        output_pixel_scale, output_pixel_scale, z, max_angle)

    # Go through and propagate between phase screens
    for i in scrnRange:

        phase = phase_screens[i]
        # print("Got phase")

        # Convert phase to radians
        phase *= phs2Rad

        # Apply phase to EField
        EFieldBuf *= numpy.exp(1j*phase)

        # Get propagation distance for this layer
        if i==(scrnNo-1):
            if ht_final == 0 and propagation_direction == "up":
                # if the final height is infinity, don't propagate any more!
                continue
            else:
                z = abs(ht_final - ht) - z_total
        else:
            z = abs(scrnAlts[i+1] - scrnAlts[i])

        # Update total distance counter
        z_total += z

        # Do ASP for last layer to next
        if max_angle is None:
            EFieldBuf[:] = opticalpropagation.angularSpectrum(
                        EFieldBuf, wavelength,
                        output_pixel_scale, output_pixel_scale, z)
        else:
            EFieldBuf[:] = opticalpropagation.limited_angularSpectrum(
                        EFieldBuf, wavelength,
                        output_pixel_scale, output_pixel_scale, z, max_angle)
        
    if(ht_final == 0) and (output_mask is not None):
        EFieldBuf *= output_mask

    return EFieldBuf

def physical_correction_propagation(
        correction, output_mask, dm_altitudes,
        wavelength, output_pixel_scale, input_efield=None, max_angle=None):
    scrnNo = len(correction)
    scrnRange = range(0, scrnNo)

    nx_output_pixels = correction[0].shape[0]

    phs2Rad = 2 * numpy.pi / (wavelength * 10 ** 9)

    EFieldBuf = input_efield
    if input_efield is None:
        EFieldBuf = numpy.exp(
                1j*numpy.zeros((nx_output_pixels,) * 2)).astype(CDTYPE)

    ht = 0
    ht_final = 0
    scrnAlts = dm_altitudes
    if output_mask is not None:
        EFieldBuf *= output_mask
    logger.debug("Create EField Buf of mask")
    
    # print([ht,scrnAlts,ht_final])

    # Propagate to first phase screen (if not already there)
    if ht != scrnAlts[0]:
        logger.debug("propagate to first phase screen")
        z = ht - scrnAlts[0]
        if max_angle is None:
            EFieldBuf[:] = opticalpropagation.angularSpectrum(
                        EFieldBuf, wavelength,
                        output_pixel_scale, output_pixel_scale, z)
        else:
            EFieldBuf[:] = opticalpropagation.limited_angularSpectrum(
                        EFieldBuf, wavelength,
                        output_pixel_scale, output_pixel_scale, z, max_angle)

    # Go through and propagate between phase screens
    for i in scrnRange:

        phase = correction[i]
        # print("Got phase")

        # Convert phase to radians
        phase *= phs2Rad

        # Apply phase to EField
        EFieldBuf *= numpy.exp(-1j*phase)

        if i==(scrnNo-1):
            z = scrnAlts[-1] - ht_final
        else:
            z = -(scrnAlts[i+1] - scrnAlts[i])

        # Do ASP for last layer to next
        if max_angle is None:
            EFieldBuf[:] = opticalpropagation.angularSpectrum(
                        EFieldBuf, wavelength,
                        output_pixel_scale, output_pixel_scale, z)
        else:
            EFieldBuf[:] = opticalpropagation.limited_angularSpectrum(
                        EFieldBuf, wavelength,
                        output_pixel_scale, output_pixel_scale, z, max_angle)
    if output_mask is not None:
        EFieldBuf *= output_mask
    
    return EFieldBuf

class ElongLineOfSight(object):
    """
    List of LineOfSight objects for elongated sources
    """

    def __init__(self, elongHeights, elongPos, config, *args, **kwargs):

        self.elongHeights = elongHeights
        self.elongPos = elongPos
        self.config = config

        self.los_list = []
        for i in range(len(self.elongHeights)):
            conf_tmp = copy.copy(config)
            conf_tmp.GSPosition = self.elongPos[i]
            conf_tmp.GSHeight = self.elongHeights[i]
            conf_tmp.calcParams()

            los = LineOfSight(conf_tmp, *args, **kwargs)
            self.los_list.append(los)

    def __getitem__(self, index):
        return self.los_list[index]

    def frame(self, scrns=None, correction=None):
        
        for i,los in enumerate(self.los_list):
            los.frame(scrns, correction)

    def zeroData(self):

        for los in self.los_list:
            los.zeroData()

    @property
    def scrns(self):
        return self.los_list[0].scrns