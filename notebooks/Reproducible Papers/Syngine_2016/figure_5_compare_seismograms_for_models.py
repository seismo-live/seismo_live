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

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# <div style='background-image: url("../../share/images/header.svg") ; padding: 0px ; background-size: cover ; border-radius: 5px ; height: 250px'>
#     <div style="float: right ; margin: 50px ; padding: 20px ; background: rgba(255 , 255 , 255 , 0.7) ; width: 50% ; height: 150px">
#         <div style="position: relative ; top: 50% ; transform: translatey(-50%)">
#             <div style="font-size: xx-large ; font-weight: 900 ; color: rgba(0 , 0 , 0 , 0.8) ; line-height: 100%">Computational Seismology</div>
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)">Reproducible Papers - Syngine Paper</div>
#         </div>
#     </div>
# </div>

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# ---
#
# # Figure 5: Seismogram Comparision
#
# This notebook is part of the supplementary materials for the Syngine paper and reproduces figure 5.
#
# This notebook creates a plot for seismograms for all models part of Syngine.
# Requires matplotlib >= 1.5 and an ObsPy version (>= 1.0) with the Syngine client.
#
# ##### Authors:
# * Lion Krischer ([@krischer](https://github.com/krischer))

# + {"deletable": true, "editable": true}
# %matplotlib inline
import obspy
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("seaborn-whitegrid")

from obspy.clients.syngine import Client
c = Client()

# + {"deletable": true, "editable": true}
# Get all available models for the Syngine service.
models = c.get_available_models()
models

# + {"deletable": true, "editable": true}
# Chile earthquake 2015-09-16 22:55:22 Mw 8.3
event_id = "GCMT:C201509162254A"

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# ```
# Date-Time (UTC): 	2015-09-16 22:55:22
# Location: 	OFF COAST OF CENTRAL CHILE
# Latitude, Longitude: 	-31.130 °, -72.090 °
# Magnitude: 	8.3 MW
# Depth: 	17.4 km
# Author: 	Global CMT Project
# Catalog, Contributor: 	GCMT, GCMT
# ```

# + {"deletable": true, "editable": true}
network = "IU"
station = "ANMO"

# Get a real station.
from obspy.clients.fdsn import Client
c_fdsn = Client("IRIS")
print(c_fdsn.get_stations(network=network, station=station, format="text")[0][0])

# + {"deletable": true, "editable": true}
# Plot the ray paths just to illustrate what we are working with.
from obspy.taup import TauPyModel
m = TauPyModel("ak135")
print(m.get_travel_times_geo(source_depth_in_km=17.4, source_latitude_in_deg=-31.130,
                             source_longitude_in_deg=-72.090, receiver_latitude_in_deg=34.95,
                             receiver_longitude_in_deg=-106.46))
m.get_ray_paths_geo(source_depth_in_km=17.4, source_latitude_in_deg=-31.130,
                    source_longitude_in_deg=-72.090, receiver_latitude_in_deg=34.95,
                    receiver_longitude_in_deg=-106.46, phase_list=["P", "pP", "sP", "PcP", "PP"]).plot();

# + {"deletable": true, "editable": true}
# Calculate distance.
from obspy.geodetics import locations2degrees

locations2degrees(lat1=-31.130, long1=-72.090, lat2=34.95, long2=-106.46)

# + {"deletable": true, "editable": true}
components = "Z"
# Varying length of databases. Use common time range.
starttime = obspy.UTCDateTime("2015-09-16T22:55:22.000000Z")
endtime = obspy.UTCDateTime("2015-09-16T23:23:46.200000Z")

data = {}
for model in models.keys():
    # Download the data for each model.
    data[model] = c.get_waveforms(model=model, network=network, station=station,
                                  components=components, dt=0.05, eventid=event_id,
                                  starttime=starttime, endtime=endtime)

# + {"deletable": true, "editable": true}
# plot everything.
plt.figure(figsize=(8, 3))
factor = 1E2

for _i, model in enumerate(sorted(data.keys(),
                                  key=lambda x: (int(x.split("_")[-1][:-1]), x))):
    tr = data[model][0]
    pos = len(data) - _i - 1
    
    plt.subplot(121)
    plt.plot(tr.times(), tr.data * factor + pos, color="0.1", lw=1.1)

    plt.yticks(list(range(9)), [""] * 9)
    plt.ylim(-0.5, 8.7)
    plt.xlim(650, 1750)
    plt.xlabel("Time since event origin [sec]")
    plt.title("Body Waves and Surface Waves Onset")
    
    plt.subplot(122)
    plt.plot(tr.times(), tr.data * factor + pos, color="0.1", lw=1.1)
    plt.yticks(list(range(9)), [""] * 9)
    plt.ylim(-0.5, 8.7)
    plt.xlim(650, 900)
    plt.text(637, pos, model, color="0.9",
             bbox=dict(facecolor="0.1", edgecolor="None"),
             ha="center", fontsize=7)
    plt.xlabel("Time since event origin [sec]")
    plt.title("First Body Waves")
    

    
plt.tight_layout()
plt.savefig("compare_seismograms_for_all_models.pdf")
plt.show()
