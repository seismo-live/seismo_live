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
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)">Part 4: Finite Sources</div>
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

# ## Advanced Exercise 2: Finite Source Effects
#
# For earthquakes with Magnitude up to about 5 recorded at teleseismic distances, approximating the fault by a point source is a reasonable approach. However, for larger earthquakes with longer rupture duration this approximation is not valid anymore. In this exercise, you will compare the point source approximation with finite source solutions to understand its limitations.
#
# For three of the earthquakes we use in this tutorial, USGS provides finite fault solutions: 
# the recent event in [Nepal](http://earthquake.usgs.gov/earthquakes/eventpage/us20002926#scientific_finitefault),
# the largest [aftershock](http://earthquake.usgs.gov/earthquakes/eventpage/us20002ejl#scientific_finitefault)
# and the one in [Chile](http://earthquake.usgs.gov/earthquakes/eventpage/usc000nzvd#scientific_finitefault). This is the fault solution and slip as a function of time for the Nepal M7.9 event:

# <p style="float: left; font-size: 9pt; text-align: center; width: 49%; margin-right: 1%; margin-bottom: 0.5em;"><img src="images/finite_source_fault.png" style="width: 100%">Fault representation (image: USGS)</p><p style="float: right; font-size: 9pt; text-align: center; width: 49%; margin-right: 1%; margin-bottom: 0.5em;"><img src="images/finite_source_stf.png" style="width: 100%">Source Time Function (image: USGS)</p><p style="clear: both;">

# -----
#
# Basic lines to set up the notebook and some paths.

# %matplotlib inline
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import obspy
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 8)

# Import Instaseis and open the database:

import instaseis
db = instaseis.open_db("data/database")

# -----

# In Instaseis, a finite fault is represented as set of point sources, where each point source represents one of the fault patches with individual source time function. This functionality is provided by the `instaseis.FiniteSource` object [(see Documentation)](http://instaseis.net/source.html#instaseis.source.FiniteSource). It can be initialized in two ways: from a list of point sources, or more conveniently by reading \*.param files provided by USGS or standard rupture format (\*.srf) files (these can also be used in the GUI).

finite_source = instaseis.FiniteSource.from_usgs_param_file(
    'data/events/finite_source/FINITE_SOURCE_2015_05_12__Mw_7_2_Nepal.param')
print(finite_source)

# A point source can be computed as a sum of all point sources weighted by their moment:

finite_source.compute_centroid()
print(finite_source.CMT)

# The hypo- and epicenter can be found as the fault patch that ruptures first:

finite_source.find_hypocenter()
print('hypocenter latitude:', finite_source.hypocenter_latitude,
      'longitude:', finite_source.hypocenter_longitude,
      'depth:', finite_source.hypocenter_depth_in_m / 1e3)

# **Task:** Compare the seismograms for three different representations of the source: 
#
# * A point source with simple gaussian source time function (using CMTSOLUTION or quakeml files),
# * the CMT solution using the more complex source time function provided by `finite_source.CMT`
# * the full finite source solution using the `FiniteSource` object and `db.get_seismograms_finite_source()`
#
# **Note:** First, you have to adapt the sampling of the source time functions in the finite source to the database, which works like this:

# +
# reloading finite source here to be sure to have a clean source time function
finite_source = instaseis.FiniteSource.from_usgs_param_file(
    'data/events/finite_source/FINITE_SOURCE_2015_05_12__Mw_7_2_Nepal.param')

# prepare the source time functions to be at the same sampling as the database
# first use enough samples such that the lowpassed stf will still be correctly represented
nsamp = int(db.info.period / finite_source[0].dt) * 50
finite_source.resample_sliprate(dt=finite_source[0].dt, nsamp=nsamp)
# lowpass to avoid aliasing
finite_source.lp_sliprate(freq=1.0/db.info.period)
# finally resample to the sampling as the database
finite_source.resample_sliprate(dt=db.info.dt, nsamp=db.info.npts)

finite_source.compute_centroid()
# -

# **Solution**

# +
# load receivers from stations xml file
receivers = instaseis.Receiver.parse('data/stations/all_stations.xml')

simple_source = instaseis.Source.parse(
    'data/events/quakeml/GCMT_2015_04_25__Mw_7_9.xml')

# compute seismogram with CMT solution and no simple source time function (gaussian):
tr_simple = 

# compute seismogram with CMT solution and source time function computed as the 
# sum of all source time functions in the finite source (reconvolve_stf=True):
tr_cmt = 

# compute seismogram for finite source
tr_finite = 

plt.plot(tr_simple.times(), tr_simple.data, label='simple')
plt.plot(...)
plt.plot(...)
 
plt.legend()
plt.xlim(0, tr_simple.times()[-1])

plt.show()
