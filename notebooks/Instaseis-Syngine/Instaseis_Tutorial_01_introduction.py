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
#             <div style="font-size: xx-large ; font-weight: 900 ; color: rgba(0 , 0 , 0 , 0.8) ; line-height: 100%">Instaseis Tutorial</div>
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)">Part 1: Introduction</div>
#         </div>
#     </div>
# </div>

# Seismo-Live: http://seismo-live.org
#
# ##### Authors:
# * Martin van Driel ([@martinvandriel](https://github.com/martinvandriel))
# * Lion Krischer ([@krischer](https://github.com/krischer))
# ---

# <img style="width:50%" src="images/logo.png">

# **Goal:** calculate seismograms on a 1D background model with [Instaseis](http://www.instaseis.net), assuming the
# Green's function database computed with [AxiSEM](http://www.axisem.info) is provided.

# ### Overview:
#
# **Introduction:**
#
# * Part 1: Very basic introduction to the API, calculate the first seismogram.
#
# **Basic Tasks:**
#
# * Part 2: Some examples of interaction with obspy, calculate synthetics for a set of events and stations
#
# **Advanced Tasks:**
#
# * Part 3: Plot record section
# * Part 4: Finite Source, compare to point source solution

# -----
#
# Basic lines to set up the notebook and some paths.

# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import os
import obspy
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 8)

# -----

# ## Basic Instaseis API Introduction
#
# Please also have a look at our webpage - http://www.instaseis.net/ - where everything is documented extensively.
#
# ### Opening a Database
#
# To get going you have to import the package `instaseis`.

import instaseis

# An Instaseis database must be opened before it can be used. These can be either a local files computed with AxiSEM, or even more easily a remote database hosted by IRIS (http://ds.iris.edu/ds/products/syngine/).

db = instaseis.open_db("syngine://prem_a_10s")

# Some basic information about the loaded database can be reviewed by just printing it.

print(db)

# From this you can already glance a couple of aspects of the database used for this tutorial:
#
# * uses anisotropic prem as its 1D model
# * is accurate for periods down to 10 seconds
# * includes vertical and horizontal components
# * sources can have depths ranging from 0 to 700 km
# * five hour long seismograms
#
# To avoid delays that become relevant when requesting very many seismograms, IRIS also offers the databases for download.

# ### Receivers and Sources
#
# Instaseis calculates seismograms for any source and receiver pair. A receiver has coordinates and optionally network and station codes. Using a reciprocal database, all receivers are assumed to be at the same depth, i.e. usually at the Earth surface.

rec = instaseis.Receiver(latitude=44.06238, longitude=10.59698,
                         network="IV", station="BDI")
print(rec)

# Sources are naturally a bit more complex and Instaseis offers a variety of ways to define them. A straightforward way for earthquakes is to pass coordinates, moment as well as strike, dip and rake.

src = instaseis.Source.from_strike_dip_rake(
    latitude=27.77, longitude=85.37, depth_in_m=12000.0,
    M0=1e+21, strike=32., dip=62., rake=90.)
print(src)

# Note that origin time was not provided and hence defaults to 1970. A non double-couple source can directly be specified in terms of its moment tensor (note that instaseis uses SI units, i.e. NM, while GCMT uses dyn cm).

src = instaseis.Source(
    latitude=27.77, longitude=85.37, depth_in_m=12000.0,
    m_rr=8.29e+20, m_tt=-2.33e+20, m_pp=-5.96e+20,
    m_rt=2.96e+20, m_rp=4.74e+20, m_tp=-3.73e+20)
print(src)

# **Sidenote:** The moment tensor can be visualized using the Beachball function from obspy.imaging:

# +
from obspy.imaging.beachball import beachball

mt = src.tensor / src.M0 # normalize the tensor to avoid problems in the plotting
beachball(mt, size=200, linewidth=2, facecolor='b');
# -

# Now we are ready to extract synthetic seismograms from the database. The return type is an [obspy stream object](https://docs.obspy.org/packages/autogen/obspy.core.stream.Stream.html), which can directly be plotted:

st = db.get_seismograms(source=src, receiver=rec, components='ZNERT')
print(st)
st.plot();

# **Done** This is all you need for a basic usage of Instaseis!
