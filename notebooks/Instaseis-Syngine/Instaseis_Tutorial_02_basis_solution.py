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
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)">Part 2: First Basic Exercise</div>
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

# ## First Basic Exercise
#
# **Task:** Calculate three component synthetic seismograms for the stations and events in the **data/events** and **data/stations** subdirectories and save them on disc.

# #### Notes
#
# 1. Receiver objects can also be created from StationXML, SEED, or STATIONS files as well as obpy inventories using `instaseis.Receiver.parse()`; see the [documentation](http://www.instaseis.net/source.html#receiver) for details.
# 2. Source objects can also be created from QuakeML, CMTSOLUTIONS, and in other ways using `instaseis.Source.parse()`; see the [documentation](http://www.instaseis.net/source.html#source) for details.
# 3. The `get_seismograms()` method has a couple of extra arguments:
#   * `kind`: `displacement`, `velocity`, `acceleration`
#   * `remove_source_shift`, `reconvolve_stf`, `dt`,
#   
#   ... see the [documentation](http://www.instaseis.net/instaseis.html#instaseis.base_instaseis_db.BaseInstaseisDB.get_seismograms) for details.
# 4. You can use the properties of the Receiver and Source objects to create usefull filenames.

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

# ### 1. Load Receivers
#
# **reminder:** you can use ObsPy to load stations and plot a map:

# +
from obspy import read_inventory

inventory = read_inventory('data/stations/all_stations.xml')
inventory.plot(projection="local", resolution="i");
# -

# This inventory can directly be used as input to generate a list of `instaseis.Receiver` objects:

receivers = instaseis.Receiver.parse(inventory)
for rec in receivers[:2]:
    print(rec)

# **Alternatively**, instaseis can directly open the station xml or STATIONS file (but then you don't have the nice plot):

receivers = instaseis.Receiver.parse('data/stations/all_stations.xml')
print(receivers[0])
receivers = instaseis.Receiver.parse('data/stations/STATIONS')
print(receivers[0])

# ### 2. Load Events
# **reminder:** use ObsPy to load events from a QuakeML file containing all events and plot a map:

# +
import glob # provides iterator to loop over files

# cat = obspy.core.event.Catalog()

for filename in glob.iglob('data/events/quakeml/*.xml'):
     cat += obspy.read_events(filename)
        
print(cat)
print(instaseis.Source.parse(cat.events[0]))
cat.plot();
# -

# **Alternatively** load QuakeML or CMTSOLUTION files directly using `instaseis.Source.parse()` and store the sources in a list:

# +
sources = []

for filename in glob.iglob('data/events/quakeml/*.xml'):
    sources.append(instaseis.Source.parse(filename))
    
print(sources[0])

for filename in glob.iglob('data/events/cmtsolutions/*'):
    sources.append(instaseis.Source.parse(filename))

print(sources[0])
# -

# ### 3. Extract Seismograms and Save to File

# For the first solution using a ObsPy event catalog:

# + {"tags": ["exercise"]}
dt = 1.0

for event in cat:
    src = instaseis.Source.parse(event)
    srcname = '%s_Mw_%3.1f' % (src.origin_time.date, src.moment_magnitude)
    for rec in receivers:
        # create a usefull filename
        recname = '%s_%s' % (rec.network, rec.station)
        filename = '%s_%s' % (recname, srcname)
        filename = filename.replace('.', '_')
        
        # extract seismograms using instaseis
        
        
        # write to miniseed files in the data_out folder. Write as MiniSEED due to multi
        # component support.
        

# + {"tags": ["solution"], "cell_type": "markdown"}
# #### Solutions:

# + {"tags": ["solution"]}
dt = 1.0

for event in cat:
    src = instaseis.Source.parse(event)
    srcname = '%s_Mw_%3.1f' % (src.origin_time.date, src.moment_magnitude)
    for rec in receivers:
        # create a usefull filename
        recname = '%s_%s' % (rec.network, rec.station)
        filename = '%s_%s' % (recname, srcname)
        filename = filename.replace('.', '_')
        
        # extract seismograms using instaseis
        st = db.get_seismograms(source=src, receiver=rec, dt=dt)
        
        # write to miniseed files in the data_out folder. Write as MiniSEED due to multi
        # component support.
        st.write(os.path.join('data_out', filename + '.mseed'), format='mseed')
# -

# For the second solution use a list of sources:

# + {"tags": ["exercise"]}
dt = 1.0

for src in sources:
    srcname = '%s_Mw_%3.1f' % (src.origin_time.date, src.moment_magnitude)
    for rec in receivers:
        # create a usefull filename
        recname = '%s_%s' % (rec.network, rec.station)
        filename = '%s_%s' % (recname, srcname)
        filename = filename.replace('.', '_')
        
        # extract seismograms using instaseis
        
        
        # write to miniseed files in the data_out folder. Write as MiniSEED due to multi
        # component support.
        

# + {"tags": ["solution"], "cell_type": "markdown"}
# #### Solutions:

# + {"tags": ["solution"]}
dt = 1.0

for src in sources:
    srcname = '%s_Mw_%3.1f' % (src.origin_time.date, src.moment_magnitude)
    for rec in receivers:
        # create a usefull filename
        recname = '%s_%s' % (rec.network, rec.station)
        filename = '%s_%s' % (recname, srcname)
        filename = filename.replace('.', '_')
        
        # extract seismograms using instaseis
        st = db.get_seismograms(source=src, receiver=rec, dt=dt)
        
        # write to miniseed files in the data_out folder. Write as MiniSEED due to multi
        # component support.
        st.write(os.path.join('data_out', filename + '.mseed'), format='mseed')
# -

# Check the data:

# ls data_out/
