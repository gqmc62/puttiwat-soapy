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
"""
The module simulating Deformable Mirrors in Soapy

DMs in Soapy
============
DMs are represented in Soapy by python objects which are initialised at startup
with some configuration parameters given, as well as a list of one or more
WFS objects which can be used to measure an interaction matrix.

Upon creation of an interaction matrix, the object first generations all the
possible independant shapes which the DM may form, known as "influence functions".
Then each influence function is passed to the specified WFS(s) and the response
noted to form an interaction matrix. The interaction matrix may then be used
to forma reconstructor.

During the AO loop, commands corresponding to the required amplitude of each
DM influence function are sent to the :py:meth:`DM.dmFrame` method, which
returns an array representing the DMs shape.

Adding New DMs
==============

New DMs are easy to add into the simulation. At its simplest, the :py:class:`DM`
class is inherited by the new DM class. Only a ``makeIMatShapes` method need be provided,
which creates the independent influence function the DM can make. The
base class deals with the rest, including making interaction matrices and loop
operation.
"""
import numpy
from scipy.ndimage.interpolation import rotate
import astropy.io.fits as fits
import aotools

from matplotlib import pyplot as plt

from aotools import karhunenLoeve as KL
from . import logger, interp, atmosphere
# from .aotools import interp, circle

ASEC2RAD = (1./3600) * (numpy.pi/180.)

try:
    xrange
except NameError:
    xrange = range

class DM(object):
    """
    The base DM class

    This class is intended to be inherited by other DM classes which describe
    real DMs. It provides methods to create DM shapes and then interaction matrices,
    given a specific WFS or WFSs.

    Parameters:
        soapy_config (ConfigObj): The soapy configuration object
        n_dm (int): The ID number of this DM
        wfss (list, optional): A list of Soapy WFS object with which to record the interaction matrix
        mask (ndarray, optional): An array or size (simConfig.simSize, simConfig.simSize) which is 1 at the telescope aperture and 0 else-where. If None then a circle is generated.
    """

    def __init__ (self, soapy_config, n_dm=0, wfss=None, mask=None, atms=None):

        # Sort out some required attributes
        self.soapy_config = soapy_config
        self.n_dm = n_dm

        self.simConfig = self.soapy_config.sim
        self.config = self.dmConfig = self.soapy_config.dms[n_dm]

        self.pupil_size = self.soapy_config.sim.pupilSize
        self.sim_size = self.soapy_config.sim.simSize
        self.scrn_size = self.soapy_config.sim.scrnSize
        self.altitude = self.config.altitude
        self.diameter = self.config.diameter
        self.telescope_diameter = self.soapy_config.tel.telDiam
        
        self.random_seed = self.config.random_seed
        
        # if self.config.type == 'Aberration':
        #     if (self.config.subtype == 'Zernike') or (self.config.subtype == 'OneZernike'):
        #         self.diameter *= numpy.sqrt(2)
        #         self.diameter += (self.sim_size - self.pupil_size)*(self.telescope_diameter/self.pupil_size)
        

        self.wfss = wfss
        self.atms = atms

        # If supplied use the mask
        if numpy.any(mask):
            self.mask = mask
        # Else we'll just make a circle
        else:
            self.mask = aotools.circle(
                    self.pupil_size/2., self.sim_size,
                    )

        # the number of phase elements at the DM altitude
        self.nx_dm_elements = int(numpy.ceil(self.pupil_size * self.diameter / self.telescope_diameter /2)*2)
        self.dm_frame = numpy.zeros((self.nx_dm_elements, self.nx_dm_elements))

        # An array of phase screen size to be observed by a line of sight
        self.dm_screen = numpy.zeros((self.scrn_size, self.scrn_size))

        # Coordinate required to fit dm size back into screen
        self.screen_coord = int(round((self.scrn_size - self.nx_dm_elements)/2.))

        self.n_acts = self.getActiveActs()
        self.actCoeffs = numpy.zeros((self.n_acts))

        # Sort out which WFS(s) observes the DM (for iMat making)
        if self.dmConfig.wfs!=None:
            try:
                # Make sure the specifed WFS actually exists
                self.wfss = [wfss[self.dmConfig.wfs]]
                self.wfs = self.wfss[0]
            except KeyError:
                raise KeyError("DM attached to WFS {}, but that WFS is not specifed in config".format(self.dmConfig.wfs))
        else:
            self.wfss = wfss

        logger.info("Making DM Influence Functions...")
        self.makeIMatShapes()

        # If using imatshapes, sclae by imat value
        if hasattr(self, "iMatShapes"):
            self.iMatShapes *= self.config.iMatValue

        # An array of values for each actuator. 1 if actuator is valid, 0 if not
        self._valid_actuators = numpy.ones((self.n_acts))


    def getActiveActs(self):
        """
        Method returning the total number of actuators used by the DM - May be overwritten in DM classes

        Returns:
            int: number of active DM actuators
        """
        return self.dmConfig.nxActuators

    def dmFrame(self, dmCommands):
        '''
        Uses DM commands to calculate the final DM shape.

        Multiplies each of the DM influence functions by the
        corresponding DM command, then sums to create the final DM shape.
        Lastly, the mean value is subtracted to avoid piston terms building up.

        Parameters:
            dmCommands (ndarray): A 1-dimensional vector of the multiplying factor of each DM influence function
            closed (bool, optional): Specifies how to great gain.
            If ``True'' (closed) then ``dmCommands'' are multiplied by gain and summed with previous commands.
            If ``False'' (open), then ``dmCommands'' are multiplied by gain, and summed withe previous commands multiplied by (1-gain).

        Returns:
            ndarray: A 2-d array with the DM shape
        '''

        self.dm_shape = self.makeDMFrame(dmCommands)
        # Remove any piston term from DM
        self.dm_shape -= self.dm_shape.mean()


        if self.dmConfig.rotation:
            dm_size = self.dm_shape.shape
            rot_dm_shape = rotate(
                self.dm_shape, self.dmConfig.rotation,
                order=self.dmConfig.interpOrder)
            rotShape = rot_dm_shape.shape
            self.dm_shape = rot_dm_shape[:,
                              rotShape[0] / 2. - dm_size[0] / 2.:
                              rotShape[0] / 2. + dm_size[0] / 2.,
                              rotShape[1] / 2. - dm_size[1] / 2.:
                              rotShape[1] / 2. + dm_size[1] / 2.
                              ]

        # Crop or pad as appropriate
        dm_size = self.dm_shape.shape[0]

        if dm_size == self.scrn_size:
            self.dm_screen = self.dm_shape

        else:
            if dm_size>self.scrn_size:
                coord = int(round(dm_size/2. - self.scrn_size/2.))
                self.dm_screen[:] = self.dm_shape[coord: -coord, coord: -coord]

            else:
                pad = int(round((self.scrn_size - dm_size)/2))
                self.dm_screen[pad:-pad, pad:-pad] = self.dm_shape

        return self.dm_screen

    def makeDMFrame(self, actCoeffs):
        dm_shape = (self.iMatShapes.T*actCoeffs.T).T.sum(0)

        return dm_shape

    def reset(self):
        self.dm_shape[:] = 0
        self.actCoeffs[:] = 0



    def makeIMatShapes(self):
        """
        Virtual method to generate the DM influence functions
        """
        # Place holder for where some influence functions could go
        self.iMatShapes = numpy.zeros((self.n_acts))

    @property
    def valid_actuators(self):
        return self._valid_actuators

    @valid_actuators.setter
    def valid_actuators(self, valid_actuators):
        self._valid_actuators = valid_actuators
        self.n_valid_actuators = self._valid_actuators.sum()

        # Sort threough imat shapes and get rid of those not valid
        self.iMatShapes = numpy.array([self.iMatShapes[i]
                                       for i in range(len(valid_actuators)) if valid_actuators[i]])


class CustomShapes(DM):
    """
    DM shape are based on a fits file.
    Interpolation is performed if required.
    """

    def makeIMatShapes(self):
        '''
        Creates all the DM shapes which are required for creating the
        interaction Matrix.

        if the shapes are not of the correct size, it interpolates them by slices
        with non valid value either NaN or 0.

        For best result, it is however preferable that no interpolation occurs.
        '''
        try:
            with open(self.config.dmShapesFilename, "rb") as shapes_file:
                dmShape = fits.getdata(shapes_file)
        except:
            raise ValueError("Can't open fits file %s" % self.config.dmShapesFilename)

        assert dmShape.shape[0] >= int(self.n_acts), 'Not enough shapes in DM shape file'
        dmShape = dmShape[:int(self.n_acts), :, :]

        if (dmShape.shape[1] != int(self.nx_dm_elements)) or\
                (dmShape.shape[2] != int(self.nx_dm_elements)):
            logger.warning("Interpolation custom DM shapes of size "
                           "({0:1.0f}, {1:1.0f}) to size of "
                           "({2:1.0f}, {2:1.0f})".format(dmShape.shape[1],
                                                         dmShape.shape[1],
                                                         self.nx_dm_elements))
            if (numpy.any(dmShape == 0)):
                dmShape[dmShape == 0] = numpy.nan
            shapes = interp.zoomWithMissingData(dmShape,
                                                (self.nx_dm_elements,
                                                 self.nx_dm_elements),
                                                 non_valid_value=numpy.nan)
        else:
            shapes = dmShape

        shapes[numpy.isnan(shapes)] = 0
        pad = self.simConfig.simPad
        self.iMatShapes = numpy.pad(
            shapes, ((0, 0), (pad, pad), (pad, pad)), mode="constant"
        ).astype("float32")


class KarhunenLoeve(DM):
    """
    A DM which corrects using a provided number of Karhunen-Loeve Polynomials
    """

    def makeIMatShapes(self):
        '''
        Creates all the DM shapes which are required for creating the
        interaction Matrix. In this case, this is a number of K-L Polynomials
        '''
        diam = self.soapy_config.tel.telDiam
        cobs = self.soapy_config.tel.obsDiam / diam

        if self.n_acts <= 500:
            nr = 64
        elif self.n_acts <= 1000:
            nr = 72
        elif self.nacts > 1000:
            nr = 128
        shapes, _, _, _ = KL.make_kl(int(self.n_acts), int(self.nx_dm_elements),
                                     ri=cobs, nr=nr, mask=True)
        pad = self.simConfig.simPad
        self.iMatShapes = numpy.pad(
            shapes, ((0, 0), (pad, pad), (pad, pad)), mode="constant"
        ).astype("float32")


class Zernike(DM):
    """
    A DM which corrects using a provided number of Zernike Polynomials
    """

    def makeIMatShapes(self):
        '''
        Creates all the DM shapes which are required for creating the
        interaction Matrix. In this case, this is a number of Zernike Polynomials
        '''

        shapes = aotools.zernikeArray(
            int(self.n_acts + 1), int(self.nx_dm_elements))[1:]

        pad = self.simConfig.simPad
        self.iMatShapes = numpy.pad(
            shapes, ((0, 0), (pad, pad), (pad, pad)), mode="constant"
        ).astype("float32")


class Piezo(DM):
    """
    A DM emulating a Piezo actuator style stack-array DM.

    This class represents a standard stack-array style DM with push-pull actuators
    behind a continuous phase sheet. The number of actuators is given in the
    configuration file.

    Each influence function is created by started with an N x N grid of zeros,
    where N is the number of actuators in one direction, and setting a single
    value to ``1``, which corresponds with a "pushed" actuator. This grid is then
    interpolated up to the ``pupilSize``, to form the shape of the DM when that
    actuator is activated. This is repeated for all actuators.
    """

    def getActiveActs(self):
        """
        Finds the actuators which will affect phase whithin the pupil to avoid
        reconstructing for redundant actuators.
        """
        activeActs = []
        xActs = self.dmConfig.nxActuators
        self.spcing = self.nx_dm_elements/float(xActs - 1)

        for x in xrange(xActs):
            for y in xrange(xActs):
                activeActs.append([x,y])
        self.valid_act_coords = numpy.array(activeActs)
        self.xActs = xActs
        return self.valid_act_coords.shape[0]


    def makeIMatShapes(self):
        """
        Generate Piezo DM influence functions

        Generates the shape of each actuator on a Piezo stack DM
        (influence functions). These are created by interpolating a grid
        on the size of the number of actuators, with only the 'poked'
        actuator set to 1 and all others set to zero, up to the required
        simulation size. This grid is actually padded with 1 extra actuator
        spacing to avoid strange edge effects.
        """

        #Create a "dmSize" - the pupilSize but with 1 extra actuator on each
        #side
        dmSize =  int(self.nx_dm_elements + 2 * numpy.round(self.spcing))

        shapes = numpy.zeros((int(self.n_acts), dmSize, dmSize), dtype="float32")

        for i in xrange(self.n_acts):
            x,y = self.valid_act_coords[i]

            # Add one to avoid the outer padding
            x+=1
            y+=1

            shape = numpy.zeros( (self.xActs+2,self.xActs+2) )
            shape[x,y] = 1

            # Interpolate up to the padded DM size
            shapes[i] = interp.zoom_rbs(shape,
                    (dmSize, dmSize), order=self.dmConfig.interpOrder)

            shapes[i] -= shapes[i].mean()

        if dmSize == self.scrn_size:
            self.dm_screen = self.dm_shape

        else:
            if dmSize > self.scrn_size:
                coord = int(round(dmSize/2. - self.scrn_size/2.))
                shapes = shapes[:, coord:-coord, coord:-coord].astype("float32")

            else:
                pad = int(round((self.scrn_size - dmSize)/2))
                shapes = numpy.pad(
                        shapes, ((0, 0), (pad,pad), (pad,pad)), mode="constant"
                        ).astype("float32")

        self.iMatShapes = shapes




class GaussStack(Piezo):
    """
    A Stack Array DM where each influence function is a 2-D Gaussian shape.

    This class represents a Stack-Array DM, similar to the :py:class:`Piezo` DM,
    where each influence function is a 2-dimensional Gaussian function. Though
    not realistic, it provides a known influence function which can be useful
    for some analysis.
    """
    def makeIMatShapes(self):
        """
        Generates the influence functions for the GaussStack DM.

        Creates a number of Guassian distributions which are centred at points
        across the pupil to act as DM influence functions. The width of the
        guassian is determined from the configuration file.
        """
        shapes = numpy.zeros((
                self.n_acts, self.nx_dm_elements, self.nx_dm_elements))

        actSpacing = self.pupil_size/(self.dmConfig.nxActuators-1)
        width = actSpacing/2.

        for i in xrange(self.n_acts):
            x,y = self.valid_act_coords[i] * actSpacing
            shapes[i] = aotools.gaussian2d(
                    self.nx_dm_elements, width, cent=(x,y))

        self.iMatShapes = shapes

        pad = self.simConfig.simPad
        self.iMatShapes = numpy.pad(
                self.iMatShapes, ((0,0), (pad,pad), (pad,pad)), mode="constant"
                )

class TT(DM):
    """
    A class representing a tip-tilt mirror.

    This can be used as a tip-tilt mirror, it features two actuators, where each
    influence function is simply a tip and a tilt.

    """

    def getActiveActs(self):
        """
        Returns the number of active actuators on the DM. Always 2 for a TT.
        """
        self.n_active_actuators = 2
        return self.n_active_actuators


    def makeIMatShapes(self):
        """
        Forms the DM influence functions, in this case just a tip and a tilt.
        """
        # Make the TT across the entire sim shape, but want it 1 to -1 across
        # pupil
        # padMax = float(self.sim_size)/self.pupil_size

        coords = numpy.linspace(
                    -1, 1, self.nx_dm_elements)

        self.iMatShapes = numpy.array(numpy.meshgrid(coords,coords))

        # Convert to arcsecs
        tt_amp = -(ASEC2RAD * self.soapy_config.tel.telDiam/2.) * 1e9

        self.iMatShapes *= tt_amp


class FastPiezo(Piezo):
    """
    A DM which simulates a Piezo DM. Faster than standard for big simulations as interpolates on each frame.
    """

    def getActiveActs(self):
        acts = super(FastPiezo, self).getActiveActs()
        self.actGrid = numpy.zeros(
                (self.dmConfig.nxActuators, self.dmConfig.nxActuators))

        # DM size is the pupil size, but withe one extra act on each side
        self.dmSize =  self.nx_dm_elements + 2 * numpy.round(self.spcing)

        return acts

    def makeIMatShapes(self):
        pass

    def makeDMFrame(self, actCoeffs):

        self.actGrid[:] = 0
        self.actGrid[(self.valid_act_coords[:, 0], self.valid_act_coords[:, 1])] = actCoeffs*self.config.iMatValue
        # plt.imshow(self.actGrid)
        # plt.colorbar()
        # plt.show()

        # Add space around edge for 1 extra act to avoid edge effects
        actGrid = numpy.pad(self.actGrid, ((1,1), (1,1)), mode="constant")

        # Interpolate to previously determined "dmSize"
        dmShape = interp.zoom_rbs(
                actGrid, self.dmSize, order=self.dmConfig.interpOrder)


        return dmShape

    @property
    def valid_actuators(self):
        return self._valid_actuators

    @valid_actuators.setter
    def valid_actuators(self, valid_actuators):
        self._valid_actuators = valid_actuators
        self.n_valid_actuators = self._valid_actuators.sum()

        # Must remove any invalid actuators
        self.valid_act_coords = numpy.array([self.valid_act_coords[i] for i in range(len(valid_actuators)) if valid_actuators[i]])


class Phase(numpy.ndarray):
    def __new__(cls, input_array, altitude=0):
        obj = numpy.asarray(input_array).view(cls)
        obj.altitude = altitude
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'info', None)

class Aberration(DM):

    def getActiveActs(self):
        """
        Returns the number of active actuators on the DM. Always 0 for aberration.
        """
        self.n_active_actuators = 0
        self.n_valid_actuators = 0
        
        if self.config.subtype == 'OneZernike':
            pass
        if self.config.subtype == 'Atmosphere':
            self.makeATM()
        if self.config.subtype == 'Zernike':
            self.makeZModes()
            self.makeNollVariances()
        if self.config.subtype == 'Zernike_no_tip_tilt':
            self.makeZModes()
            self.makeNollVariances()
        if self.config.subtype == 'MagicDM':
            self.makeATM()
        if self.config.subtype == 'MagicDM_no_tip_tilt':
            self.makeATM()
            self.makeTipTiltPistonPupil()
            
        if self.config.subtype == 'Atmosphere_perfect_DM':
            xActs = self.dmConfig.nxActuators
            self.spcing = self.nx_dm_elements/float(xActs - 1)
            self.makeATM()
        if self.config.subtype == 'Atmosphere_perfect_DM_no_tip_tilt':
            xActs = self.dmConfig.nxActuators
            self.spcing = self.nx_dm_elements/float(xActs - 1)
            self.makeATM()
            self.makeTipTiltPistonPupil()
        
        return self.n_active_actuators
    
    def makeDMFrame(self,actCoeffs='flat'):
        if self.config.r0 == numpy.inf:
            self.aberration = numpy.zeros((self.nx_dm_elements,self.nx_dm_elements),dtype=float)
            return self.aberration
        if actCoeffs == 'flat':
            self.aberration = numpy.zeros((self.nx_dm_elements,self.nx_dm_elements),dtype=float)
            return self.aberration
        if self.config.subtype == 'OneZernike':
            self.aberration = self.makeOneZernikeAberration(actCoeffs=actCoeffs)
        if self.config.subtype == 'Atmosphere':
            self.aberration = self.makeAtmosphereAberration(actCoeffs=actCoeffs)
        if self.config.subtype == 'Zernike':
            self.aberration = self.makeZernikeAberration(actCoeffs=actCoeffs)
        if self.config.subtype == 'Zernike_no_tip_tilt':
            self.aberration = self.makeZernikeAberration_no_tip_tilt(actCoeffs=actCoeffs)
        if self.config.subtype == 'Atmosphere_perfect_DM':
            self.aberration = self.makeAtmosphereAberration_perfect_DM(actCoeffs=actCoeffs)
        if self.config.subtype == 'Atmosphere_perfect_DM_no_tip_tilt':
            self.aberration = self.makeAtmosphereAberration_perfect_DM_no_tip_tilt(actCoeffs=actCoeffs)
        if self.config.subtype == 'MagicDM':
            self.aberration = self.makeMagicDM(actCoeffs=actCoeffs)
        if self.config.subtype == 'MagicDM_no_tip_tilt':
            self.aberration = self.makeMagicDM_no_tip_tilt(actCoeffs=actCoeffs)
        
        return self.aberration
    
    def makeTipTiltPistonPupil(self):
        self.PUPIL = aotools.zernike.zernike_noll(1, self.nx_dm_elements, 0)
        self.PISTON = self.PUPIL / (self.PUPIL.flatten().T@self.PUPIL.flatten())**0.5
        self.TIP =  aotools.zernike.zernike_noll(2, self.nx_dm_elements, 0)
        self.TILT = aotools.zernike.zernike_noll(3, self.nx_dm_elements, 0)
        self.TIP /= (self.TIP.flatten().T@self.TIP.flatten())**0.5
        self.TILT /= (self.TILT.flatten().T@self.TILT.flatten())**0.5
        return
    
    def makeZModes(self):
        self.Z_modes = aotools.zernike.zernike_noll(self.config.nollMode - 1, self.nx_dm_elements, 0)
        return
    
    def makeNollVariances(self):
        self.noll_variances = numpy.zeros((self.config.nollMode - 1 - 2),dtype=float)
        for i in numpy.arange(self.config.nollMode - 1 - 2):
            self.noll_variances[i] = get_noll_variance(
                self.diameter,self.config.r0,self.config.L0,i + 4)
        return
    
    def makeATM(self):
        # print(self.nx_dm_elements, self.telescope_diameter/self.pupil_size,
        #     self.config.r0, self.config.L0)
        self.atm_ab = atmosphere.InfinitePhaseScreen(
            self.nx_dm_elements, self.telescope_diameter/self.pupil_size,
            self.config.r0, self.config.L0, wind_speed=0,
            time_step=0, wind_direction=0, random_seed=self.random_seed,
            n_columns=2)
        for row in numpy.arange(self.nx_dm_elements):                    
            self.atm_ab.add_row()
        if self.random_seed is not None:
            self.random_seed = numpy.random.randint(0,2**31)
    
    
    
    
    def makeOneZernikeAberration(self,actCoeffs='shape'):
        print('need to check this function: one zernike')
        self.noll_variance = get_noll_variance(
            self.diameter,self.config.r0,self.config.L0,self.config.nollMode)
        
        if (actCoeffs == 'shape') and (self.config.nollMode != 1):
            numpy.random.seed(self.random_seed)
            self.aberrationStrength = numpy.random.normal(0,self.noll_variance**0.5)
            if self.random_seed is not None:
                self.random_seed = numpy.random.randint(0,2**31)
            aberration = aotools.zernike.zernike_noll(self.config.nollMode, self.nx_dm_elements, 0)
            aberration *= (numpy.sqrt((numpy.pi * self.nx_dm_elements**2 / 4.)))
            aberration *= self.aberrationStrength#[*numpy.sqrt(numpy.sqrt(2)**(5./3.))
            self.aberration = aberration
            return self.aberration
        else:
            self.aberration = numpy.zeros((self.nx_dm_elements,self.nx_dm_elements),dtype=float)
            return self.aberration
    
    def makeZernikeAberration_no_tip_tilt(self,actCoeffs=0):
        print('need to check this function: zernike no tip tilt')
            
        # print(self.noll_variances,aberrationStrengths)
        # aberrationStrength = self.noll_variance**0.5
        if (actCoeffs == 'shape') and (self.config.nollMode != 1):
            numpy.random.seed(self.random_seed)
            self.aberrationStrength = numpy.random.normal(numpy.zeros((self.config.nollMode - 1 - 2)),
                                                         self.noll_variances**0.5)
            if self.random_seed is not None:
                self.random_seed = numpy.random.randint(0,2**31)
            
            self.aberration = numpy.zeros((self.nx_dm_elements,self.nx_dm_elements), dtype=float)
            

            for i in numpy.arange(self.config.nollMode - 1 - 2):               
                #temp /= (numpy.sqrt((numpy.pi * self.nx_dm_elements**2 / 4.)))
                # temp /= (self.telescope_diameter/self.pupil_size)
                self.aberration = self.Z_modes * self.aberrationStrength[i]#[*numpy.sqrt(numpy.sqrt(2)**(5./3.))
                # plt.imshow(temp);plt.title('mode:{:d}'.format(i+4));plt.show()
                self.aberration *= (500./(2.*numpy.pi))
            # try:
            #     self.plotted
            # except:
            #     # if self.config.type == 'Aberration':
            #     # plt.imshow(numpy.angle(numpy.exp(1j*aberration)))
            #     plt.imshow(aberration)
            #     plt.colorbar()
            #     plt.show()
            #     self.plotted = True
            
            return self.aberration
        else:
            self.aberration = numpy.zeros((self.nx_dm_elements,self.nx_dm_elements),dtype=float)
            return self.aberration
        
    def makeZernikeAberration(self,actCoeffs=0):
        print('need to check this function: zernike')
        if (actCoeffs == 'shape') and (self.config.nollMode != 1):
            numpy.random.seed(self.random_seed)
            self.aberrationStrength = numpy.random.normal(numpy.zeros((self.config.nollMode - 1 )),#- 2)),
                                                             self.noll_variances**0.5)
            if self.random_seed is not None:
                self.random_seed = numpy.random.randint(0,2**31)
                
            self.aberration = numpy.zeros((self.nx_dm_elements,self.nx_dm_elements), dtype=float)

            self.aberration = ((self.aberrationStrength
                               * self.Z_modes.reshape(self.config.nollMode - 1 ,#- 2,
                                                      self.nx_dm_elements
                                                      *self.nx_dm_elements).T).T.sum(0)
                               ).reshape(self.nx_dm_elements,self.nx_dm_elements)
            self.aberration *= (500./(2.*numpy.pi))
                
            
            return self.aberration
        else:
            self.aberration = numpy.zeros((self.nx_dm_elements,self.nx_dm_elements),dtype=float)
            return self.aberration
    
    def makeAtmosphereAberration_no_tip_tilt(self,actCoeffs=0):
        print('need to check this function : atmos no tip tilt')
        if (actCoeffs == 'shape'):
            aberration = numpy.copy(self.atm_ab.scrn) * (500/(2*numpy.pi))
            aberration *= self.PUPIL
            piston = aberration.flatten().T@self.PISTON.flatten()
            tip = aberration.flatten().T@self.TIP.flatten()
            tilt = aberration.flatten().T@self.TILT.flatten()
            aberration -= (piston*self.PISTON + tip*self.TIP + tilt*self.TILT)
            aberration *= self.PUPIL
            self.aberration = aberration
            return self.aberration
        else:
            self.aberration = numpy.zeros((self.nx_dm_elements,self.nx_dm_elements),dtype=float)
            return self.aberration
    
    def makeAtmosphereAberration_perfect_DM_no_tip_tilt(self,actCoeffs=0):
        # print('need to check this function: perfect dm no tip tilt')
        if (actCoeffs == 'shape'):
            aberration = numpy.copy(self.atm_ab.scrn) * (500/(2*numpy.pi))
            dmpitch = self.spcing*self.telescope_diameter/self.pupil_size
            aberration = perfect_dm(aberration,self.telescope_diameter/self.pupil_size,dmpitch)
            aberration *= self.PUPIL
            piston = aberration.flatten().T@self.PISTON.flatten()
            tip = aberration.flatten().T@self.TIP.flatten()
            tilt = aberration.flatten().T@self.TILT.flatten()
            aberration -= (piston*self.PISTON + tip*self.TIP + tilt*self.TILT)
            aberration *= self.PUPIL
            self.aberration = aberration
            return self.aberration
        else:
            self.aberration = numpy.zeros((self.nx_dm_elements,self.nx_dm_elements),dtype=float)
            return self.aberration
        
    def makeMagicDM_no_tip_tilt(self,actCoeffs=0):
        print('need to check this function: magic no tip tilt')
        if (actCoeffs == 'shape'):
        
            aberration = numpy.copy(self.atm_ab.scrn) * (500/(2*numpy.pi))
            dmpitch = self.spcing*self.telescope_diameter/self.pupil_size
            # dmpitch = self.config.r0
            aberration = perfect_dm(aberration,self.telescope_diameter/self.pupil_size,dmpitch)
            aberration *= self.PUPIL
            piston = aberration.flatten().T@self.PISTON.flatten()
            tip = aberration.flatten().T@self.TIP.flatten()
            tilt = aberration.flatten().T@self.TILT.flatten()
            aberration -= (piston*self.PISTON + tip*self.TIP + tilt*self.TILT)
            aberration *= self.PUPIL
            self.aberration = aberration
            return self.aberration
        else:
            self.aberration = numpy.zeros((self.nx_dm_elements,self.nx_dm_elements),dtype=float)
            return self.aberration
    
    def makeAtmosphereAberration(self,actCoeffs=0):
        print('need to check this function: atmos')
        if (actCoeffs == 'shape'):
            aberration = numpy.copy(self.atm_ab.scrn) * (500/(2*numpy.pi))
            self.aberration = aberration
            return self.aberration
        else:
            self.aberration = numpy.zeros((self.nx_dm_elements,self.nx_dm_elements),dtype=float)
            return self.aberration
    
    def makeAtmosphereAberration_perfect_DM(self,actCoeffs=0):
        print('need to check this function: perfect dm')
        if (actCoeffs == 'shape'):
            aberration = numpy.copy(self.atm_ab.scrn) * (500/(2*numpy.pi))
            dmpitch = self.spcing*self.telescope_diameter/self.pupil_size
            aberration = perfect_dm(aberration,self.telescope_diameter/self.pupil_size,dmpitch)
            self.aberration = aberration
            return self.aberration
        else:
            self.aberration = numpy.zeros((self.nx_dm_elements,self.nx_dm_elements),dtype=float)
            return self.aberration
        
        
    def makeMagicDM(self,actCoeffs=0):
        print('need to check this function: magic')
        if (actCoeffs == 'shape'):
            self.atm_ab = self.atms.infinite_phase_screens[self.config.conjugateLayer]
            aberration = numpy.copy(self.atm_ab.scrn) * (500/(2*numpy.pi))
            dmpitch = self.spcing*self.telescope_diameter/self.pupil_size
            aberration = perfect_dm(aberration,self.telescope_diameter/self.pupil_size,dmpitch)
            self.aberration = aberration
            return self.aberration
        else:
            self.aberration = numpy.zeros((self.nx_dm_elements,self.nx_dm_elements),dtype=float)
            return self.aberration
        
    

def perfect_dm(phase,delta,dm_pitch):
    N = phase.shape[0]
    PHASE = numpy.pad(phase,(N//2,N//2),'symmetric')
    NN = N*2
    del_f = 1./(2.*NN*delta)
    spatial_frequency = numpy.fft.fftshift(numpy.fft.fft2(PHASE))
    f = (numpy.arange(NN) - NN/2)*del_f
    fy, fx = numpy.meshgrid(f,f)
    FILTER = ((numpy.abs(fx) < 1./2./dm_pitch) * (numpy.abs(fy) < 1./2./dm_pitch))
    DM_PHASE = (numpy.fft.ifft2(numpy.fft.fftshift(spatial_frequency * FILTER)).real)[N//2:3*N//2,N//2:3*N//2]
    return DM_PHASE
def get_noll_variance(D,r0,L0,mode):
    TABLE = [0.4480, 0.0230, 0.0062, 0.0025, 0.0012]
    n, _ = aotools.functions.zernike.zernIndex(mode)
    # D = self.telescope_diameter
    # r0 = self.config.r0
    if n == 0:
        return numpy.nan # don't know what to use 6.88 - 1.0299 ?
    elif n == 1:
        # L0 = self.config.L0
        coeff = TABLE[n-1]
        A = D/L0
        correction = (1 - 1.42*A**(1./3.) + 3.70*A**2
                      - 4.1*A**(7./3.) + 4.21*A**4 - 4.00*A**(13./3.))
        noll_variance = coeff * (D/r0)**(5./3.) * correction
    elif n <= len(TABLE):
        coeff = TABLE[n - 1]
        noll_variance = coeff * (D/r0)**(5./3.)
    else:
        J = mode**(-numpy.sqrt(3)/2) - (mode + 1)**(-numpy.sqrt(3)/2)
        noll_variance = 0.2944 * J * (D/r0)**(5./3.)
    return noll_variance
        