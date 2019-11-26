# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# <div style='padding: 0px ; background-size: cover ; border-radius: 5px ; height: 250px'>
#     <div style="float: right ; margin: 50px ; padding: 20px ; background: rgba(255 , 255 , 255 , 0.7) ; width: 50% ; height: 150px">
#         <div style="position: relative ; top: 50% ; transform: translatey(-50%)">
#             <div style="font-size: xx-large ; font-weight: 900 ; color: rgba(0 , 0 , 0 , 0.8) ; line-height: 100%">General Seismology</div>
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)"> Free Oscillations: Instrument response and spectral analysis</div>
#         </div>
#     </div>
# </div>

# Seismo-Live: http://seismo-live.org
#
#
# ##### Authors:
# * Carl Tape ([@carltape](https://github.com/carltape))
# * Yongki Andita Aiman
# * Tomy Gunawan
# * Angel Ling
# ---

# ## Earth's Free Oscilation
#

# ### What are Free Oscillations?

# Like a strechted string on a guitar, an elastic sphere supports standing waves. In seismology, these are known as free osicllations or normal modes. They are characterized by discrete spectrum. Only specific frequencies are permitted, corresponding to integral numbers of wavelengths in the 3 orthogonal directions $(r,θ,Φ)$. These oscillations correspond to standing surface waves of the longest possible wavelength and lowest frequency (periods up to about 1 hour).
#
# The elastic-gravitational response of Earth to localized, transient excitations consists of impulsive disturbances followed by dispersed wave trains and long-lasting standing waves.
#
# As long as an earthquake rupture is ongoing, the Earth responds with forced vibrations. Once the rupture has ceased, Earth undergoes free oscillations around its new equilibrium state. The rupture during of the largest earthquakes are on the order of few minutes and thus very much shorter than the typical decay time of modes of a few tens of hours.  
#
# The deviations of Earth structure from a spherically symmetric reference state are quite small. It is therefore convenient to discuss free oscillations of a spherically averaged Earth and treat any devviation away from this state with perturbation theory. On a spherically symmetric Earth three integer quantum numbers, $n, l $, and $  m$, fully specify the set of normal modes. The azumuthal order, $|m|$, counts the number of nodal surfacesin the longitudinal direction $\hat\phi $. The number of nodal surfaces in the colatitudinal direction, $\hat\phi$ is $|l-m|$, where $l$ is the angular order. For fixed $l$ and $m$ the over number $n$ indexes the modes with increasing frequency.
#
#
# Solutions of the linearized, homogeneous equations of motion for a self-gravitating elastic Earth can be written as ${}^{}_{n}u^{m}_{l}(r,t) = Re[(\hat{r}_{n}U_{l}(r)Y^{m}_{l}(\theta, \phi) + {}^{}_{n}V^{}_{l}(r)\nabla_{1}Y^{m}_{l}(\theta, \phi) - {}^{}_{n}W^{}_{l}(r)\hat{r} \times \nabla_{1}Y^{m}_{l}(\theta, \phi))e^{i_{n}\omega^{m}_{l}t}]$
#
# where  ${}^{}_{n}u^{m}_{l}$ is the displacement eigenfucntion of the mode singlet identified by the $(n, l, m)$- triplet and $\omega^{m}_{l}$ is its eigenfrequency. $Y^{m}_{l}$ are surface spherical harmonic functions and $\Delta_{1}$ is the surface gradient operator. While $n$ and $l$ can be any non-negative number $(n \geq 0, l \geq 0)$, the azimuthal order is limited
# to the interval $-l \leq m \leq l$. The spherical harmonics describe the angular shape of the eigenfunction. The three scalar radial eigenfunctions, ${}_{n}U_{l}$, ${}_{n}V_{l}(r)$ and ${}_{n}W_{l}$  describe the way the mode samples Earth with depth. As $n$ increases they become more oscillatory with depth leading to an increased number of nodal spheres.    
#

# Following are some low-order spherical harmonics. 

# ![High%20resolution%20spherical%20harmonics.png](attachment:High%20resolution%20spherical%20harmonics.png)

# Surface spherical harmonics $Y^{m}_{l} = X^{m}_{l}(\theta)e^{{i}m{\phi}}$ that compose the basis set for $ l = 0, 1, and 2$ spheroidal modes (e.g., ${}_{0}S_{1}$, ${}_{2}S_{1}$, ${}_{0}S_{2}$, ${}_{3}S_{2}$) plotted in Hammer–Aitoff projection. The singlets of modes have these shapes on the $\hat{r}$ component of recordings on a spherical, nonrotating Earth. 

# ### Very First Free Oscillations and Computations!
# The longest period oscillations are only excited in a measurable way by the largest earthquakes, and it was not until the 1960 Chile earthquake that they were unambiguously identified on long-period seismograms. 
#
# Freeman Gilbert first wrote computer program EOS to solve the ordinary differential equations governing free oscillations, and various descendants of this code have been circulating informally since the early 1970s. We mention two versions here: OBANI by John Woodhouse and MINOS by Guy Masters. Woodhouse advanced the code by allowing to compute the eigenfuctions through the method of minors. He also introduced a mode counter for spheroidal modes, while Masters added one for toroidal and radial, Stoneley and IC modes.

# ### Types of Free Oscillations!
#
# The complete set of normal modes forms a basis for the description of any general elastic displacement that can occur within the Earth, and this property is used to calculate theoretical seismograms for long-period surface waves.
# Similar to the separation of Love and Rayleigh surface waves, the normal modes separate into two distinct types of oscillation:
# 1. Spheroidal - Corresponds to standing Rayleigh waves
# 2. Toroidal - Corresponds to standing Love waves
#
#
#

# ### Surface waves
# They can be understood as a superposition of free oscillations. Excluding very deep earthquakes, fundamental mode surface waves are the largest signal in a seismogram. Surface wave packets are relatively short and do not require the consistently high signal levels, over several days, as normal modes do. Analysis of surfaces involves the analysis of fundamental modes and the first few overtones, at high frequencies. 
# In the upper mantle, imaging capabilities using body waves are dictated by the distribution of earthquakes and seismic stations. Surface waves travel along the surface between sources and recievers, crossing remote areas and thereby picking up invaluable information about along-path upper-mantle structure that remains elusive to body waves. 

# ![seismograph.png](attachment:seismograph.png)

# So which wiggles are the earthquake? The P wave will be the first wiggle that is bigger than the rest of the little ones (the microseisms). Because P waves are the fastest seismic waves, they will usually be the first ones that your seismograph records. The next set of seismic waves on your seismogram will be the S waves. These are usually bigger than the P waves. If there aren't any S waves marked on your seismogram, it probably means the earthquake happened on the other side of the planet. S waves can't travel through the liquid layers of the earth so these waves never made it to your seismograph.
#
# The surface waves (Love and Rayleigh waves) are the other, often larger, waves marked on the seismogram. They have a lower frequency, which means that waves (the lines; the ups-and-downs) are more spread out. Surface waves travel a little slower than S waves (which, in turn, are slower than P waves) so they tend to arrive at the seismograph just after the S waves. For shallow earthquakes (earthquakes with a focus near the surface of the earth), the surface waves may be the largest waves recorded by the seismograph. Often they are the only waves recorded a long distance from medium-sized earthquakes.

# ### Few permanent Seismic Network across the world!
#  US american Global Seismic Network (GSN) operated by IRIS (Incorporated Research Institution for Seismology)
#
#
# which includes 
#
# 1. High-Gain Long-Period Network
# 2. World-Wide Standardized Seismograph Network - WWSN 
# 3. China Digital Seismograph Network - CDSN 
#
#
# GEOSCOPE from France and GEOFON from Germany are other global networks
#

# ### Exercise: 

# + {"code_folding": [0]}
# Import Libraries (PLEASE RUN THIS CODE FIRST!)
# Make sure to execute this cell first!
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')                  # do not show warnings
from IPython.display import display, Math, Latex
from time import *
from obspy import *
from obspy.core import read, UTCDateTime
from obspy.clients.fdsn import Client
import numpy as np
import matplotlib.pylab as plt
from obspy.signal.filter import lowpass
from matplotlib.mlab import detrend
from obspy.signal.invsim import cosine_taper 
from mpl_toolkits.basemap import Basemap
from obspy.imaging.beachball import beach
plt.style.use('ggplot')
# -

# ### And identify few features of Seismographs using real data from below!!

# Analyzing the raw 10-day time series of the Sumatra earthquake and Earthquake at East Coast of Honshu, Japan recorded at station CAN (Canberra, Australia), which is shown in Figure 1. We will use the notation c(t) (c is “counts”) to represent the time series.
# 1. What is the duration of the seismogram?
# 2. What is the time step?
# 3. What is the Nyquist frequency?
# 4. Zoom in on the y-limits with **ylim(3e4*[-1 1])** What is the approximate period (in seconds and in days) of the most conspicuous oscillation?
# 5. Find the largest aftershock in the second half of the record. Use the example code to extract the absolute time, then list the location, magnitude, and origin time of the event from a catalog search in www.globalcmt.org. Hint: the magnitude is Mw > 6.
# 6. Does this aftershock have the same mechanism as the mainshock?
# 7. In summary, identify and interpret the main features in the seismogram. Hint: noise is a feature.

# ### For Sumatra - 2004-12-26T00:58:53

# + {"code_folding": [0]}
# Getting the waveforms from IRIS data center in USA
client = Client("IRIS")
t = UTCDateTime("2004-12-26T00:58:53.0")   # origin time of the earthquake
starttime = t-(24*3600)                    # one day before the origin time
endtime = t+(9*24*3600)                    # 9 days after the origin time
st = client.get_waveforms("*", "CAN", "*", "LHZ", starttime, endtime, attach_response=True)

# + {"code_folding": [0]}
# Take a copy of the stream to avoid overwriting the original data
stream = st.copy()
tr = stream[0]

# Choosing the 1st stream, Specify sampling parameters and Nyquist frequency
npts = stream[0].stats.npts           # number of samples
df = stream[0].stats.sampling_rate    # sampling rate
nsec = npts/df                        # sampling time
timestep = 1/df                       # time step
fNy = df / 2.0                        # Nyquist frequency
dur_sec = npts/df                     # Duration of the seismogram in second
dur_day = dur_sec/(24*60*60)          # Duration of the seismogram in day
time = np.linspace(0,nsec,(nsec*df))  # time axis for plotting

print ("1. The duration of the seismogram is", dur_sec, "seconds or", dur_day, "days")
print ("2. The time step is", timestep, "s")
print ("3. The Nyquist frequency is", fNy, "Hz")
print ("4. Zoom-in on the y-axis, showing that the record is dominated by aftershocks (spikes) and a ~12-hour period of tides.")

# + {"code_folding": [0]}
# Plotting
plt.rcParams['figure.figsize'] = 15, 5
plt.rcParams['lines.linewidth'] = 0.5

plt.plot(time,tr.data)
#print (starttime)                    #identify the start time of the seismogram
plt.title('CAN[LHZ], starting time:2004-12-25T00:58:53.000000Z')
plt.xlabel('Time [s]')
plt.ylabel('Counts')
plt.ylim(-3E4,3E4)
plt.show()
# -

# 5.
# The largest aftershock occurs around 2005-01-01 06:49:32. GCMT catalog reveals that this is a Mw 6.7 aftershock:
#
# * 200501010625A OFF W COAST OF NORTHERN 
# * Date: 2005/ 1/ 1 Centroid Time: 6:25:48.3 GMT
# * Lat= 4.97 Lon= 92.22
# * Depth= 12.0 Half duration= 5.2
# * Centroid time minus hypocenter time: 3.5 
# * Moment Tensor: Expo=26 -0.052 -0.669 0.721 -0.287 -0.378 -0.854 
# * Mw = 6.7 mb = 6.0 Ms = 6.7 Scalar Moment = 1.2e+26 
# * Fault plane: strike=111 dip=68 slip=-173 
# * Fault plane: strike=18 dip=84 slip=-22 

# ### Exercise 2:  Do the same For Honshu, Japan - 2011-03-11T05:46:24

# + {"code_folding": [0]}
# Getting the waveforms from IRIS data center in USA
client = Client("IRIS")
t = UTCDateTime("2011-03-11T05:46:24")   # origin time of the earthquake
starttime = t-(24*3600)                    # one day before the origin time
#starttime = UTCDateTime("2011-02-11T05:46:24")
endtime = t+(9*24*3600)                    # 9 days after the origin time
#endtime =  UTCDateTime("2011-12-11T05:46:24")
pre_filt = (0.5, 0.6, 3.0, 5.0)
st = client.get_waveforms("*", "CAN", "*", "LHZ", starttime, endtime, attach_response=True)



# + {"code_folding": [0]}
# Take a copy of the stream to avoid overwriting the original data
stream = st.copy()
tr = stream[0]

# Choosing the 1st stream, Specify sampling parameters and Nyquist frequency
npts = stream[0].stats.npts           # number of samples
df = stream[0].stats.sampling_rate    # sampling rate
nsec = npts/df                        # sampling time
timestep = 1/df                       # time step
fNy = df / 2.0                        # Nyquist frequency
dur_sec = npts/df                     # Duration of the seismogram in second
dur_day = dur_sec/(24*60*60)          # Duration of the seismogram in day
time = np.linspace(0,nsec,(nsec*df))  # time axis for plotting

print ("1. The duration of the seismogram is", dur_sec, "seconds or", dur_day, "days")
print ("2. The time step is", timestep, "s")
print ("3. The Nyquist frequency is", fNy, "Hz")
print ("4. Zoom-in on the y-axis, showing that the record is dominated by aftershocks (spikes) and a ~12-hour period of tides.")

# + {"code_folding": [0]}
# Plotting
plt.rcParams['figure.figsize'] = 15, 5
plt.rcParams['lines.linewidth'] = 0.5

plt.plot(time,tr.data)
#print (starttime)                    #identify the start time of the seismogram
plt.title('CAN[LHZ], starting time:2011-03-11T05:46:24.000000Z')
plt.xlabel('Time [s]')
plt.ylabel('Counts')
plt.ylim(-3E4,3E4)
plt.show()
# -

# There are two ways to prevent overamplification while convolving the inverted instrument spectrum: One possibility is to specify a water level which represents a clipping of the inverse spectrum and limits amplification to a certain maximum cut-off value (water_level in dB). The other possibility is to taper the waveform data in the frequency domain prior to multiplying with the inverse spectrum, i.e. perform a pre-filtering in the frequency domain (specifying the four corner frequencies of the frequency taper as a tuple in pre_filt).

# + {"code_folding": [0]}
#initialization
from obspy import UTCDateTime
from obspy.clients.fdsn import Client


t = UTCDateTime("2011-03-11T05:46:24")   # origin time of the earthquake
starttime = t                   # one day before the origin time
#starttime = UTCDateTime("2011-02-11T05:46:24")
endtime = t+(1*24*3600)                    # 9 days after the origin time
#endtime =  UTCDateTime("2011-12-11T05:46:24")
fdsn_client = Client('IRIS')
# Fetch waveform from IRIS FDSN web service into a ObsPy stream object
# and automatically attach correct response
st = client.get_waveforms("*", "CAN", "*", "LHZ", starttime, endtime, attach_response=True)



    



# + {"code_folding": [0]}
# Take a copy of the stream to avoid overwriting the original data
stream = st.copy()
tr = stream[0]
pre_filt = [0.001, 0.005, 45, 50]
tr.remove_response(pre_filt=pre_filt, output='VEL',
                   water_level=60, plot=True) 

# Choosing the 1st stream, Specify sampling parameters and Nyquist frequency
npts = stream[0].stats.npts           # number of samples
df = stream[0].stats.sampling_rate    # sampling rate
nsec = npts/df                        # sampling time
timestep = 1/df                       # time step
fNy = df / 2.0                        # Nyquist frequency
dur_sec = npts/df                     # Duration of the seismogram in second
dur_day = dur_sec/(24*60*60)          # Duration of the seismogram in day
time = np.linspace(0,nsec,(nsec*df))  # time axis for plotting

print ("1. The duration of the seismogram is", dur_sec, "seconds or", dur_day, "days")
print ("2. The time step is", timestep, "s")
print ("3. The Nyquist frequency is", fNy, "Hz")
print ("4. Zoom-in on the y-axis, showing that the record is dominated by aftershocks (spikes) and a ~12-hour period of tides.")
# -

# ### Spectrum of Vertical Component
# The displacement at any point on the surface of the Earth can be quite complicated but can be thought of as a sum of discrete modes of oscillation, each mode having a characteristic frequency and decay rate which are dependent upon the structure of the Earth. The initial amplitudes of the m rodes of free oscillation depend upon the source of excitation which, in free-oscillation siesmology is usually an earthquake. For examle, a mangitude 6.5 earthquake will rupture for about ten seconds after which the Earth is in free oscillation. Away from the immediate vicinity of the earthquake, the motions of the Earth are small in amplitude and total displacement at a recording site can be written as a sum of decaying cosinusoids. 
#
#
#
#  $u(t) = \sum_{k}A_{k}cos(\omega_{k}t + \phi_{k})e^{-\alpha_{k}t}  $
#  
# where $\omega_{k}$ is the frequency of $k$'th mode which has an initial amplitude $A_{k}$ and an initial phase $\phi_{k}$. $\alpha_{k}$ controls the decay rate of the $k$'th mode and is often written in terms of the "quality" of the mode, $Q_{k}$ where $ Q_{k} = \frac{\omega{k}}{2\alpha_{k}} $ In the Earth, attenuation of seismic energy is weak and the $Q$'s of modes are typically between 100 and 6000. 
#
# Long wavelengths are typically associated with low frequencies and free oscillation theory has usually been used to describe motions of the Earth with periods between about 100 seconds and 1 hour. This period corresponds to mode ${}_{0}S^{}_{2}$ sometimes reffered as "football mode". 
#
#
#
#

# ### Passing the seismic wave through a high pass filter to observe vertical components
#

# + {"code_folding": [0]}
# Getting the waveforms from IRIS data center in USA
client = Client("IRIS")
t = UTCDateTime("2011-03-11T05:46:24")   # origin time of the earthquake
starttime = t-(24*3600)                    # one day before the origin time
#starttime = UTCDateTime("2011-02-11T05:46:24")
endtime = t+(9*24*3600)                    # 9 days after the origin time
#endtime =  UTCDateTime("2011-12-11T05:46:24")
pre_filt = (0.5, 0.6, 3.0, 5.0)
st = client.get_waveforms("*", "CAN", "*", "LHZ", starttime, endtime, attach_response=True)
st.filter("lowpass", freq=0.08)

# + {"code_folding": [0]}
# Take a copy of the stream to avoid overwriting the original data
stream = st.copy()
tr = stream[0]


# Choosing the 1st stream, Specify sampling parameters and Nyquist frequency
npts = stream[0].stats.npts           # number of samples
df = stream[0].stats.sampling_rate    # sampling rate
nsec = npts/df                        # sampling time
timestep = 1/df                       # time step
fNy = df / 2.0                        # Nyquist frequency
dur_sec = npts/df                     # Duration of the seismogram in second
dur_day = dur_sec/(24*60*60)          # Duration of the seismogram in day
time = np.linspace(0,nsec,(nsec*df))  # time axis for plotting

print ("1. The duration of the seismogram is", dur_sec, "seconds or", dur_day, "days")
print ("2. The time step is", timestep, "s")
print ("3. The Nyquist frequency is", fNy, "Hz")
print ("4. Zoom-in on the y-axis, showing that the record is dominated by aftershocks (spikes) and a ~12-hour period of tides.")

# + {"code_folding": [0]}
# Plotting
plt.rcParams['figure.figsize'] = 15, 5
plt.rcParams['lines.linewidth'] = 0.5

plt.plot(time,tr.data)
#print (starttime)                    #identify the start time of the seismogram
plt.title('CAN[LHZ], starting time:2011-03-11T05:46:24.000000Z')
plt.xlabel('Time [s]')
plt.ylabel('Counts')
plt.ylim(-3E4,3E4)
plt.show()

# + {"code_folding": [0]}
# Take a copy of the stream to avoid overwriting the original data
y = st.copy()[0].data                         

# Taper and Detrend Signal
taper_percentage = 0.1                                              # Percentage of tapering applied to signal
y_td = detrend((y* cosine_taper(npts,taper_percentage)), 'linear')  # Taper signal

# Frequency Domain
y_fft = np.fft.rfft(y_td) 

# Zeros padding
y_pad = np.lib.pad(y_td, (92289,92289),'constant',constant_values=(0,0))
y_fftpad= np.fft.rfft(y_pad)

# + {"code_folding": []}
# Plotting parameter
plt.rcParams['figure.figsize'] = 10, 5
plt.rcParams['lines.linewidth'] = 0.5

# Plot Frequency spectrum in [0.2, 1] mHz window
freq = np.linspace(0, fNy, len(y_fft))  
freq = [x*1000 for x in freq]
plt.plot(freq, abs(y_fft), 'r', lw=1.5) 
plt.legend()
plt.title('Frequency Spectrum')
plt.xlabel('Frequency [mHz]')
plt.ylabel('Amplitude[counts]')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlim(0.2, 1.0) # limited to frequency window
plt.ylim(0.0, 1.5E7) # Zoom in 
plt.show()

# + {"code_folding": [0]}
# Plotting parameter
plt.rcParams['figure.figsize'] = 10, 5
plt.rcParams['lines.linewidth'] = 0.5
f0 = 1/(24*3600)
color = ['b','y','k','g','m','c']
k = [0.31, 0.47, 0.64, 0.82 , 0.94]
label = ['0.3mHz', '0.47mHz', '0.64mHz', '0.82mHz', '0.94mHz']

# Plot Frequency spectrum in [0.2, 1] mHz window
freq = np.linspace(0, fNy, len(y_fft))  
freq = [x*1000 for x in freq]
plt.plot(freq, abs(y_fft), 'r', lw=1.5)
for i in range(len(k)):
    c = plt.axvline(k[i], color=color[i], label=label[i], lw=2)
plt.legend()
plt.title('Frequency Spectrum')
plt.xlabel('Frequency [mHz]')
plt.ylabel('Amplitude[counts]')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.xlim(0.2, 1.0) # limited to frequency window
plt.ylim(0.0, 1.5E7) # Zoom in 
plt.show()
# -


