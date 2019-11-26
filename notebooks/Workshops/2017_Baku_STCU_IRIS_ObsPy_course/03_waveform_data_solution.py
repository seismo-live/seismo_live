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

# <div style='background-image: url("share/baku.jpg") ; padding: 0px ; background-size: cover ; border-radius: 15px ; height: 250px; background-position: 0% 80%'>
#     <div style="float: right ; margin: 50px ; padding: 20px ; background: rgba(255 , 255 , 255 , 0.9) ; width: 50% ; height: 150px">
#         <div style="position: relative ; top: 50% ; transform: translatey(-50%)">
#             <div style="font-size: xx-large ; font-weight: 900 ; color: rgba(0 , 0 , 0 , 0.9) ; line-height: 100%">ObsPy Tutorial</div>
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.7)">Handling Waveform Data</div>
#         </div>
#     </div>
# </div>
# image: User:Abbaszade656 / Wikimedia Commons / <a href="http://creativecommons.org/licenses/by-sa/4.0/">CC-BY-SA-4.0</a>

# ## Workshop for the "Training in Network Management Systems and Analytical Tools for Seismic"
# ### Baku, October 2018
#
# Seismo-Live: http://seismo-live.org
#
# ##### Authors:
# * Tobias Megies ([@megies](https://github.com/megies))
# * Lion Krischer ([@krischer](https://github.com/krischer))
# ---

# ![](images/obspy_logo_full_524x179px.png)

# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 12, 8

# <img src="images/Stream_Trace.svg" width=90%>
#
# * read waveform data is returned as a **`Stream`** object.

from obspy import read
st = read("./data/waveform_PFO.mseed", format="mseed")
print(st)

# - UNIX wildcards can be used to read multiple files simultaneously
# - automatic file format detection, no need to worry about file formats
#
#   - currently supported: **MiniSEED, SAC, SEG-Y, SEG-2, GSE1/2, Seisan, SH, DataMark, CSS, wav, y, Q, ASCII, Guralp Compressed Format, Kinemetrics EVT, K-NET&KiK formats, PDAS, Reftek 130, WIN (keeps growing...)**
#   - more file formats are included whenever a basic reading routine is provided (or e.g. sufficient documentation on data compression etc.)
#   - custom user-specific file formats can be added (through plug-ins) to be auto-discovered in a local ObsPy installation

from obspy import read
st = read("./data/waveform_*")
print(st)

# - for MiniSEED files, only reading short portions of files without all of the file getting read into memory is supported (saves time and memory when working on large collections of big files)

# +
from obspy import UTCDateTime

t = UTCDateTime("2011-03-11T05:46:23.015400Z")
st = read("./data/waveform_*", starttime=t + 10 * 60, endtime=t + 12 * 60)
print(st)
# -

# #### The Stream Object
#
#  - A Stream object is a collection of Trace objects

from obspy import read
st = read("./data/waveform_PFO.mseed")
print(type(st))

print(st)

print(st.traces)

print(st[0])

# - More traces can be assembled using **`+`** operator (or using the `.append()` and `.extend()` methods)

# +
st1 = read("./data/waveform_PFO.mseed")
st2 = read("./data/waveform_PFO_synthetics.mseed")

st = st1 + st2
print(st)

st3 = read("./data/waveform_BFO_BHE.sac")

st += st3
print(st)
# -

#  - convenient (and nicely readable) looping over traces

for tr in st:
    print(tr.id)

#  - Stream is useful for applying the same processing to a larger number of different waveforms or to group Traces for processing (e.g. three components of one station in one Stream)

# #### The Trace Object
#
# - a Trace object is a single, contiguous waveform data block (i.e. regularly spaced time series, no gaps)
# - a Trace object contains a limited amount of metadata in a dictionary-like object (as **`Trace.stats`**) that fully describes the time series by specifying..
#   * recording location/instrument (network, station, location and channel code)
#   * start time
#   * sampling rate

st = read("./data/waveform_PFO.mseed")
tr = st[0]  # get the first Trace in the Stream
print(tr)

print(tr.stats)

# - For custom applications it is sometimes necessary to directly manipulate the metadata of a Trace.
# - The metadata of the Trace will **stay consistent**, as all values are derived from the starttime, the data and the sampling rate and are **updated automatically**

print(tr.stats.delta, "|", tr.stats.endtime)

tr.stats.sampling_rate = 5.0
print(tr.stats.delta, "|", tr.stats.endtime)

print(tr.stats.npts)

tr.data = tr.data[:100]
print(tr.stats.npts, "|", tr.stats.endtime)

# - convenience methods make basic manipulations simple

tr = read("./data/waveform_PFO.mseed")[0]
tr.plot()

print(tr)
tr.resample(sampling_rate=100.0)
print(tr)

print(tr)
tr.trim(tr.stats.starttime + 12 * 60, tr.stats.starttime + 14 * 60)
print(tr)
tr.plot()

tr.detrend("linear")
tr.taper(max_percentage=0.05, type='cosine')
tr.filter("lowpass", freq=0.1)
tr.plot()

# try tr.<Tab> for other methods defined for Trace
tr.detrend?

# - Raw data available as a [**`numpy.ndarray`**](http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html) (as **`Trace.data`**)

print(tr.data[:20])

# - Data can be directly modified e.g. ..
#
# ..by doing arithmetic operations (fast, handled in C by NumPy)

print(tr.data ** 2 + 0.5)

# ..by using [**`numpy.ndarray`** builtin methods](http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html) (also done in C by NumPy)

print(tr.data.max())
print(tr.data.mean())
print(tr.data.ptp())
# try tr.data.<Tab> for a list of numpy methods defined on ndarray

# ..by using **`numpy`** functions (also done in C by NumPy)

import numpy as np
print(np.abs(tr.data))
# you can try np.<Tab> but there is a lot in there
# try np.a<Tab>

# ..by feeding pointers to existing C/Fortran routines from inside Python!
#
# This is done internally in several places, e.g. for cross correlations, beamforming or in third-party filetype libraries like e.g. libmseed.
#
# - Trace objects can also be manually generated with data in a [**`numpy.ndarray`**](http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html) (e.g. when needing to parse waveforms from non-standard ascii files)

# +
from obspy import Trace

x = np.random.randint(-100, 100, 500)
tr = Trace(data=x)
tr.stats.station = "XYZ"
tr.stats.starttime = UTCDateTime()

tr.plot()
# -

# - Stream objects can be assembled from manually generated Traces

# +
from obspy import Stream

tr2 = Trace(data=np.random.randint(-300, 100, 1000))
tr2.stats.starttime = UTCDateTime()
tr2.stats.sampling_rate = 10.0
st = Stream([tr, tr2])

st.plot()
# -

# #### Builtin methods defined on **`Stream`** / **`Trace`**
#
# - Most methods that work on a Trace object also work on a Stream object. They are simply executed for every trace. [See ObsPy documentation for an overview of available methods](http://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.html) (or try **`st.<Tab>`**).
#  - **`st.filter()`** - Filter all attached traces.
#  - **`st.trim()`** - Cut all traces.
#  - **`st.resample()`** / **`st.decimate()`** - Change the sampling rate.
#  - **`st.trigger()`** - Run triggering algorithms.
#  - **`st.plot()`** / **`st.spectrogram()`** - Visualize the data.
#  - **`st.attach_response()`**/**`st.remove_response()`**, **`st.simulate()`** - Instrument correction
#  - **`st.merge()`**, **`st.normalize()`**, **`st.detrend()`**, **`st.taper()`**, ...
# - A **`Stream`** object can also be exported to many formats, so ObsPy can be used to convert between different file formats.

st = read("./data/waveform_*.sac")
st.write("output_file.mseed", format="MSEED")
# !ls -l output_file*

# <img src="images/Stream_Trace.svg" width=90%>

# #### Trace Exercises
#  - Make an **`numpy.ndarray`** with zeros and (e.g. use **`numpy.zeros()`**) and put an ideal pulse somewhere in it
#  - initialize a **`Trace`** object with your data array
#  - Fill in some station information (e.g. network, station, ..)
#  - Print trace summary and plot the trace
#  - Change the sampling rate to 20 Hz
#  - Change the starttime of the trace to the start time of this session
#  - Print the trace summary and plot the trace again

x = np.zeros(300)
x[100] = 1.0
tr = Trace(data=x)
tr.stats.station = "ABC"
tr.stats.sampling_rate = 20.0
tr.stats.starttime = UTCDateTime(2014, 2, 24, 15, 0, 0)
print(tr)
tr.plot()

# - Use **`tr.filter(...)`** and apply a lowpass filter with a corner frequency of 1 Hertz.
# - Display the preview plot, there are a few seconds of zeros that we can cut off.

tr.filter("lowpass", freq=1)
tr.plot()

# - Use **`tr.trim(...)`** to remove some of the zeros at start and at the end
# - show the preview plot again

tr.trim(tr.stats.starttime + 3, tr.stats.endtime - 5)
tr.plot()

# - Scale up the amplitudes of the trace by a factor of 500
# - Add standard normal gaussian noise to the trace (use [**`np.random.randn()`**](http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.randn.html))
# - Display the preview plot again

tr.data = tr.data * 500
tr.data = tr.data + np.random.randn(len(tr))
tr.plot()

# #### Stream Exercises
#
# - Read all Tohoku example earthquake data into a stream object ("./data/waveform\_\*")
# - Print the stream summary

st = read("./data/waveform_*")
print(st)

# - Use **`st.select()`** to only keep traces of station BFO in the stream. Show the preview plot.

st = st.select(station="BFO")
st.plot()

# - trim the data to a 10 minute time window around the first arrival (just roughly looking at the preview plot)
# - display the preview plot and spectrograms for the stream (with logarithmic frequency scale, use `wlen=50` for the spectrogram plot)

t1 = UTCDateTime(2011, 3, 11, 5, 55)
st.trim(t1, t1 + 10 * 60)
st.plot()
st.spectrogram(log=True, wlen=50);

# - remove the linear trend from the data, apply a tapering and a lowpass at 0.1 Hertz
# - show the preview plot again

st.detrend("linear")
st.filter("lowpass", freq=0.1)
st.plot()
