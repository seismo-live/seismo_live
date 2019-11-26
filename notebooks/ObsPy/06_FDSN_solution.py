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

# <div style='background-image: url("../share/images/header.svg") ; padding: 0px ; background-size: cover ; border-radius: 5px ; height: 250px'>
#     <div style="float: right ; margin: 50px ; padding: 20px ; background: rgba(255 , 255 , 255 , 0.7) ; width: 50% ; height: 150px">
#         <div style="position: relative ; top: 50% ; transform: translatey(-50%)">
#             <div style="font-size: xx-large ; font-weight: 900 ; color: rgba(0 , 0 , 0 , 0.8) ; line-height: 100%">ObsPy Tutorial</div>
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)">Downloading Data</div>
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

# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 12, 8

# ObsPy has clients to directly fetch data via...
#
# - FDSN webservices (IRIS, Geofon/GFZ, USGS, NCEDC, SeisComp3 instances, ...)
# - ArcLink (EIDA, ...)
# - Earthworm
# - SeedLink (near-realtime servers)
# - NERIES/NERA/seismicportal.eu
# - NEIC
# - SeisHub (local seismological database)
#
# This introduction shows how to use the FDSN webservice client. The FDSN webservice definition is by now the default web service implemented by many data centers world wide. Clients for other protocols work similar to the FDSN client.
#
# #### Waveform Data

# +
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

client = Client("IRIS")
t = UTCDateTime("2011-03-11T05:46:23")  # Tohoku
st = client.get_waveforms("II", "PFO", "*", "LHZ",
                          t + 10 * 60, t + 30 * 60)
print(st)
st.plot()
# -

# - again, waveform data is returned as a Stream object
# - for all custom processing workflows it does not matter if the data originates from a local file or from a web service
#
# #### Event Metadata
#
# The FDSN client can also be used to request event metadata:

t = UTCDateTime("2011-03-11T05:46:23")  # Tohoku
catalog = client.get_events(starttime=t - 100, endtime=t + 24 * 3600,
                            minmagnitude=7)
print(catalog)
catalog.plot();

# Requests can have a wide range of constraints (see [ObsPy Documentation](http://docs.obspy.org/packages/autogen/obspy.fdsn.client.Client.get_events.html)):
#
# - time range
# - geographical (lonlat-box, circular by distance)
# - depth range
# - magnitude range, type
# - contributing agency

# #### Station Metadata
#
# Finally, the FDSN client can be used to request station metadata. Stations can be looked up using a wide range of constraints (see [ObsPy documentation](http://docs.obspy.org/packages/autogen/obspy.fdsn.client.Client.get_stations.html)):
#
#  * network/station code
#  * time range of operation
#  * geographical (lonlat-box, circular by distance)

# +
event = catalog[0]
origin = event.origins[0]

# Münster
lon = 7.63
lat = 51.96

inventory = client.get_stations(longitude=lon, latitude=lat,
                                maxradius=2.5, level="station")
print(inventory)
# -

# The **`level=...`** keyword is used to specify the level of detail in the requested inventory
#
# - `"network"`: only return information on networks matching the criteria
# - `"station"`: return information on all matching stations
# - `"channel"`: return information on available channels in all stations networks matching the criteria
# - `"response"`: include instrument response for all matching channels (large result data size!)

inventory = client.get_stations(network="OE", station="DAVA",
                                level="station")
print(inventory)

inventory = client.get_stations(network="OE", station="DAVA",
                                level="channel")
print(inventory)

# For waveform requests that include instrument correction, the appropriate instrument response information can be attached to waveforms automatically:     
# (Of course, for work on large datasets, the better choice is to download all station information and avoid the internal repeated webservice requests)

# +
t = UTCDateTime("2011-03-11T05:46:23")  # Tohoku
st = client.get_waveforms("II", "PFO", "*", "LHZ",
                          t + 10 * 60, t + 30 * 60, attach_response=True)
st.plot()

st.remove_response()
st.plot()
# -

# All data requested using the FDSN client can be directly saved to file using the **`filename="..."`** option. The data is then stored exactly as it is served by the data center, i.e. without first parsing by ObsPy and outputting by ObsPy.

client.get_events(starttime=t-100, endtime=t+24*3600, minmagnitude=7,
                  filename="/tmp/requested_events.xml")
client.get_stations(network="OE", station="DAVA", level="station",
                    filename="/tmp/requested_stations.xml")
client.get_waveforms("II", "PFO", "*", "LHZ", t + 10 * 60, t + 30 * 60,
                     filename="/tmp/requested_waveforms.mseed")
# !ls -lrt /tmp/requested*

# #### FDSN Client Exercise
#
# Use the FDSN client to assemble a waveform dataset for on event.
#
# - search for a large earthquake (e.g. by depth or in a region of your choice, use option **`limit=5`** to keep network traffic down)

# + {"tags": ["exercise"]}


# + {"tags": ["solution"]}
from obspy.clients.fdsn import Client

client = Client()
catalog = client.get_events(minmagnitude=7, limit=5, mindepth=400)
print(catalog)
catalog.plot()
event = catalog[0]
print(event)
# -

# - search for stations to look at waveforms for the event. stations should..
#     * be available at the time of the event
#     * have a vertical 1 Hz stream ("LHZ", to not overpower our network..)
#     * be in a narrow angular distance around the event (e.g. 90-91 degrees)
#     * adjust your search so that only a small number of stations (e.g. 3-6) match your search criteria

# + {"tags": ["exercise"]}


# + {"tags": ["solution"]}
origin = event.origins[0]
t = origin.time

inventory = client.get_stations(longitude=origin.longitude, latitude=origin.latitude,
                                minradius=101, maxradius=102,
                                starttime=t, endtime =t+100,
                                channel="LHZ", matchtimeseries=True)
print(inventory)
# -

# - for each of these stations download data of the event, e.g. a couple of minutes before to half an hour after the event
# - put all data together in one stream (put the `get_waveforms()` call in a try/except/pass block to silently skip stations that actually have no data available)
# - print stream info, plot the raw data

# + {"tags": ["exercise"]}


# + {"tags": ["solution"]}
from obspy import Stream
st = Stream()

for network in inventory:
    for station in network:
        try:
            st += client.get_waveforms(network.code, station.code, "*", "LHZ",
                                       t - 5 * 60, t + 30 * 60, attach_response=True)
        except:
            pass

print(st)
st.plot()
# -

# - correct the instrument response for all stations and plot the corrected data

# + {"tags": ["exercise"]}


# + {"tags": ["solution"]}
st.remove_response(water_level=20)
st.plot()
# -

# If you have time, assemble and plot another similar dataset (e.g. like before stations at a certain distance from a big event, or use Transportable Array data for a big event, etc.)
