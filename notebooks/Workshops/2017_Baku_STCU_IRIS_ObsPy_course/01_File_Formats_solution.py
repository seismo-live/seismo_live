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
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.7)">Introduction to file formats and reading/writing data with ObsPy</div>
#         </div>
#     </div>
# </div>
# image: User:Abbaszade656 / Wikimedia Commons / <a href="http://creativecommons.org/licenses/by-sa/4.0/">CC-BY-SA-4.0</a>

# ## Workshop for the "Training in Network Management Systems and Analytical Tools for Seismic"
# ### Baku, October 2018
#
#
# Seismo-Live: http://seismo-live.org
#
# ##### Authors:
# * Lion Krischer ([@krischer](https://github.com/krischer))
# * Tobias Megies ([@megies](https://github.com/megies))
# ---

# ![](images/obspy_logo_full_524x179px.png)

# While oftentimes not taught, it is important to understand the types of data available in seismology, at least at a basic level. The notebook at hand teaches you how to use different types of seismological data in ObsPy.

# **This notebook aims to give a quick introductions to ObsPy's core functions and classes. Everything here will be repeated in more detail in later notebooks.**

# Using ObsPy revolves around three central function's and the objects they return. Three types of data are classically distinguished in observational seismology and each of these map to one function in ObsPy:
#
# * `obspy.read()`: Reads waveform data to `obspy.Stream` and `obspy.Trace` objects.
# * `obspy.read_inventory()`: Reads station information to `obspy.Inventory` objects.
# * `obspy.read_events()`: Reads event data to `obspy.Catalog` objects.
#
# The specific format of each of these types of data is automatically determined, each function supports reading from URLs, various compression formats, in-memory files, and other sources. Many different file formats can also be written out again. The resulting objects enable the manipulation of the data in various ways.
#
# One of the main goals of ObsPy is to free researchers from having to worry about which format their data is coming in to be able to focus at the task at hand.

# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 12, 8

# ## FDSN/SEED Identifiers
#
# According to the  [SEED standard](www.fdsn.org/seed_manual/SEEDManual_V2.4.pdf), which is fairly well adopted, the following nomenclature is used to identify seismic receivers:
#
# * **Network identifier**: Identifies the network/owner of the data. Assigned by the FDSN and thus unique.
# * **Station identifier**: The station within a network. *NOT UNIQUE IN PRACTICE!* Always use together with a network code!
# * **Location identifer**: Identifies different data streams within one station. Commonly used to logically separate multiple instruments at a single station.
# * **Channel identifier**: Three character code: 1) Band and approximate sampling rate, 2) The type of instrument, 3) The orientation
#
# This results in full ids of the form **NET.STA.LOC.CHA**, e.g. **BK.HELL.00.BHZ**. *(ObsPy uses this ordering, putting the station identifier first is also fairly common in other packages.)*
#
#
# ---
#
#
# In seismology we generally distinguish between three separate types of data:
#
# 1. **Waveform Data** - The actual waveforms as time series.
# 2. **Station Data** - Information about the stations' operators, geographical locations, and the instruments' responses.
# 3. **Event Data** - Information about earthquakes and other seismic sources.
#
# Some formats have elements of two or more of these.

# ## Waveform Data
#
# ![stream](images/Stream_Trace.svg)
#
# There are a myriad of waveform data formats, but in Europe and the USA, two formats dominate: **MiniSEED** and **SAC**
#
#
# ### MiniSEED
#
# * This is what you get from datacenters and usually also what they store, thus the original data
# * Most useful as a streaming and archival format
# * Can store integers and single/double precision floats
# * Integer data (e.g. counts from a digitizer) are heavily compressed: a factor of 3-5 depending on the data
# * Can deal with gaps and overlaps
# * Multiple components per file
# * Contains only the really necessary parameters and some information for the network operators and data providers

# +
# To use ObsPy, you always have to import it first.
import obspy

# ObsPy automatically detects the file format. ALWAYS
# (no matter the data format) results in Stream object.
st = obspy.read("data/example.mseed")

# Printing an object usually results in some kind of
# informative string.
print(st)
# -

# Format specific information is an attribute named
# after the format.
print(st[0].stats.mseed)

# Use the .plot() method for a quick preview plot.
st.plot()

# +
# This is a quick interlude to teach you the basics of how to work
# with Stream/Trace objects.

# Most operations work in-place, e.g. they modify the existing
# objects. This is done for performance reasons and to match
# typical seismological workflows.

# We'll create a copy here - otherwise multiple executions of this
# notebook cell will keep modifying the data - one of the caveats
# of the notebooks.
st2 = st.copy()

# To use only parts of a Stream, use the select() function.
print(st2.select(component="Z"))

# Stream objects behave like a list of Trace objects.
tr = st2[0]

# Plotting also works for single traces.
tr.plot()

# Some basic processing. Please note that these modify the
# existing object.
tr.detrend("linear")
tr.taper(type="hann", max_percentage=0.05)
tr.filter("lowpass", freq=0.05, corners=4)

# Plot again.
tr.plot()
# -

# You can write it again by simply specifing the format.
st.write("temp.mseed", format="mseed")

# ### SAC
#
# * Custom format of the `SAC` code.
# * Simple header and single precision floating point data.
# * Only a single component per file and no concept of gaps/overlaps.
# * Used a lot due to `SAC` being very popular and the additional basic information that can be stored in the header.
# * ObsPy internally uses a `SACTrace` object to do this which can also be directly used if full control over SAC files is desired/necessary.

st = obspy.read("data/example.sac")
print(st)
# SAC specific information is stored in
# the .sac attribute.
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
# * `(dataless)SEED`: Very complete but a complex binary format. Still used a lot, e.g. for the Arclink protocol
# * `RESP`: A strict subset of SEED. ASCII based. Contains **ONLY** the response.
# * `StationXML`: Essentially like SEED but cleaner and based on XML. Most modern format and what the datacenters nowadays serve. **Use this if you can.**
#
#
# ObsPy can work with all of them in the same fashion, but today we will focus on StationXML.

# They are XML files:

# !head data/all_stations.xml

# +
import obspy

# Use the read_inventory function to open them. This function
# will return Inventory objects.
inv = obspy.read_inventory("data/all_stations.xml")
print(inv)
# -

# You can see that they can contain an arbitrary number of networks, stations, and channels.

# ObsPy is also able to plot a map of them.
inv.plot(projection="local");

# As well as a plot the instrument response.
inv.select(network="BK", station="HELL").plot_response(0.001);

# Coordinates of single channels can also be extraced. This function
# also takes a datetime arguments to extract information at different
# points in time.
inv.get_coordinates("BK.HELL.00.BHZ")

# And it can naturally be written again, also in a modified state.
inv.select(channel="BHZ").write("temp.xml", format="stationxml")

# ## Event Data
#
# ![events](./images/Event.svg)
#
# Event data is often served in very simple formats like NDK or the CMTSOLUTION format used by many waveform solvers:

# !cat data/GCMT_2014_08_24__Mw_6_1

# Datacenters on the hand offer **QuakeML** files, which are surprisingly complex in structure but can store detailed relations between pieces of data.

# Read QuakeML files with the read_events() function.
# cat = obspy.read_events("data/GCMT_2014_08_24__Mw_6_1.xml")
print(cat)

print(cat[0])

cat.plot(projection="ortho");

# Once again they can be written with the write() function.
cat.write("temp_quake.xml", format="quakeml")

# To show off some more things, I added a file containing all events the USGS has for a three degree radius around Livermore for the last three years.

# +
import obspy

# cat = obspy.read_events("data/last_three_years_around_livermore.zmap")

print(cat)
# -

cat.plot(projection="local", resolution="i");

cat.filter("magnitude > 3")
