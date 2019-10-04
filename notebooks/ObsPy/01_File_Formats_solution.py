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

# <div style='background-image: url("../share/images/header.svg") ; padding: 0px ; background-size: cover ; border-radius: 5px ; height: 250px'>
#     <div style="float: right ; margin: 50px ; padding: 20px ; background: rgba(255 , 255 , 255 , 0.7) ; width: 50% ; height: 150px">
#         <div style="position: relative ; top: 50% ; transform: translatey(-50%)">
#             <div style="font-size: xx-large ; font-weight: 900 ; color: rgba(0 , 0 , 0 , 0.8) ; line-height: 100%">ObsPy Tutorial</div>
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)">Introduction to File Formats and read/write in ObsPy</div>
#         </div>
#     </div>
# </div>

# Seismo-Live: http://seismo-live.org
#
# ##### Authors:
# * Lion Krischer ([@krischer](https://github.com/krischer))
# * Tobias Megies ([@megies](https://github.com/megies))
# ---

# ![](images/obspy_logo_full_524x179px.png)

# This is oftentimes not taught, but fairly important to understand, at least at a basic level. This also teaches you how to work with these in ObsPy.

# **This notebook aims to give a quick introductions to ObsPy's core functions and classes. Everything here will be repeated in more detail in later notebooks.**

# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 12, 8

# ## SEED Identifiers
#
# According to the  [SEED standard](www.fdsn.org/seed_manual/SEEDManual_V2.4.pdf), which is fairly well adopted, the following nomenclature is used to identify seismic receivers:
#
# * **Network code**: Identifies the network/owner of the data. Assigned by the FDSN and thus unique.
# * **Station code**: The station within a network. *NOT UNIQUE IN PRACTICE!* Always use together with a network code!
# * **Location ID**: Identifies different data streams within one station. Commonly used to logically separate multiple instruments at a single station.
# * **Channel codes**: Three character code: 1) Band and approximate sampling rate, 2) The type of instrument, 3) The orientation
#
# This results in full ids of the form **NET.STA.LOC.CHAN**, e.g. **IV.PRMA..HHE**.
#
#
# ---
#
#
# In seismology we generally distinguish between three separate types of data:
#
# 1. **Waveform Data** - The actual waveforms as time series.
# 2. **Station Data** - Information about the stations' operators, geographical locations, and the instrument's responses.
# 3. **Event Data** - Information about earthquakes.
#
# Some formats have elements of two or more of these.

# ## Waveform Data
#
# ![stream](images/Stream_Trace.svg)
#
# There are a myriad of waveform data formats but in Europe and the USA two formats dominate: **MiniSEED** and **SAC**
#
#
# ### MiniSEED
#
# * This is what you get from datacenters and also what they store, thus the original data
# * Can store integers and single/double precision floats
# * Integer data (e.g. counts from a digitizer) are heavily compressed: a factor of 3-5 depending on the data
# * Can deal with gaps and overlaps
# * Multiple components per file
# * Contains only the really necessary parameters and some information for the data providers

# +
import obspy

# ObsPy automatically detects the file format.
st = obspy.read("data/example.mseed")
print(st)

# Fileformat specific information is stored here.
print(st[0].stats.mseed)
# -

st.plot()

# +
# This is a quick interlude to teach you basics about how to work
# with Stream/Trace objects.

# Most operations work in-place, e.g. they modify the existing
# objects. We'll create a copy here.
st2 = st.copy()

# To use only part of a Stream, use the select() function.
print(st2.select(component="Z"))

# Stream objects behave like a list of Trace objects.
tr = st2[0]

tr.plot()

# Some basic processing. Please note that these modify the
# existing object.
tr.detrend("linear")
tr.taper(type="hann", max_percentage=0.05)
tr.filter("lowpass", freq=0.5)

tr.plot()
# -

# You can write it again by simply specifing the format.
st.write("temp.mseed", format="mseed")

# ### SAC
#
# * Custom format of the `sac` code.
# * Simple header and single precision floating point data.
# * Only a single component per file and no concept of gaps/overlaps.
# * Used a lot due to `sac` being very popular and the additional basic information that can be stored in the header.

st = obspy.read("data/example.sac")
print(st)
st[0].stats.sac.__dict__

st.plot()

# You can once again write it with the write() method.
st.write("temp.sac", format="sac")

# ## Station Data
#
# ![inv](images/Inventory.svg)
#
# Station data contains information about the organziation that collections the data, geographical information, as well as the instrument response. It mainly comes in three formats:
#
# * `(dataless) SEED`: Very complete but pretty complex and binary. Still used a lot, e.g. for the Arclink protocol
# * `RESP`: A strict subset of SEED. ASCII based. Contains **ONLY** the response.
# * `StationXML`: Essentially like SEED but cleaner and based on XML. Most modern format and what the datacenters nowadays serve. **Use this if you can.**
#
#
# ObsPy can work with all of them but today we will focus on StationXML.

# They are XML files:

# !head data/all_stations.xml

# +
import obspy

# Use the read_inventory function to open them.
inv = obspy.read_inventory("data/all_stations.xml")
print(inv)
# -

# You can see that they can contain an arbirary number of networks, stations, and channels.

# ObsPy is also able to plot a map of them.
inv.plot(projection="local");

# As well as a plot the instrument response.
inv.select(network="IV", station="SALO", channel="BH?").plot_response(0.001);

# Coordinates of single channels can also be extraced. This function
# also takes a datetime arguments to extract information at different
# points in time.
inv.get_coordinates("IV.SALO..BHZ")

# And it can naturally be written again, also in modified state.
inv.select(channel="BHZ").write("temp.xml", format="stationxml")

# ## Event Data
#
# ![events](./images/Event.svg)
#
# Event data is essentially served in either very simple formats like NDK or the CMTSOLUTION format used by many waveform solvers:

# !cat data/GCMT_2014_04_01__Mw_8_1

# Datacenters on the hand offer **QuakeML** files, which are surprisingly complex in structure but can store complex relations.

# Read QuakeML files with the read_events() function.
# cat = obspy.read_events("data/GCMT_2014_04_01__Mw_8_1.xml")
print(cat)

print(cat[0])

cat.plot(projection="ortho");

# Once again they can be written with the write() function.
cat.write("temp_quake.xml", format="quakeml")

# To show off some more things, I added a file containing all events from 2014 in the GCMT catalog.

# +
import obspy

# cat = obspy.read_events("data/2014.ndk")

print(cat)
# -

cat.plot();

cat.filter("depth > 100000", "magnitude > 7")
