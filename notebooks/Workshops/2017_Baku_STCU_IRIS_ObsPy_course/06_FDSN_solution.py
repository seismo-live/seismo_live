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
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.7)">Downloading Data</div>
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
import obspy
from obspy.clients.fdsn import Client

client = Client("IRIS")
t = obspy.UTCDateTime("2014-08-24T10:20:44.0")  # South Napa Earthquake
st = client.get_waveforms("II", "PFO", "00", "LHZ",
                          t - 5 * 60, t + 20 * 60)
print(st)
st.plot()
# -

# - again, waveform data is returned as a Stream object
# - for all custom processing workflows it does not matter if the data originates from a local file or from a web service
#
# #### Event Metadata
#
# The FDSN client can also be used to request event metadata:

t = obspy.UTCDateTime("2014-08-24T10:20:44.0")  # South Napa earthquake
catalog = client.get_events(starttime=t - 100, endtime=t + 3600,
                            minmagnitude=6)
print(catalog)
catalog.plot();

# Requests can have a wide range of constraints (see [ObsPy Documentation](http://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_events.html)):
#
# - time range
# - geographical (lonlat-box, circular by distance)
# - depth range
# - magnitude range, type
# - contributing agency

# #### Station Metadata
#
# Finally, the FDSN client can be used to request station metadata. Stations can be looked up using a wide range of constraints (see [ObsPy documentation](http://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_stations.html)):
#
#  * network/station code
#  * time range of operation
#  * geographical (lonlat-box, circular by distance)

# +
event = catalog[0]
origin = event.origins[0]

# Livermore
lon = -121.768005
lat = 37.681873

# Get currently active stations in 5 km radius around Livermore.
inventory = client.get_stations(longitude=lon, latitude=lat,
                                maxradius=0.05, level="station",
                                starttime=obspy.UTCDateTime())
print(inventory)
inventory.plot(projection="local", resolution="i");
# -

# The **`level=...`** keyword is used to specify the level of detail in the requested inventory
#
# - `"network"`: only return information on networks matching the criteria
# - `"station"`: return information on all matching stations
# - `"channel"`: return information on available channels in all stations networks matching the criteria
# - `"response"`: include instrument response for all matching channels (large result data size!)

inventory = client.get_stations(network="II", station="PFO",
                                level="station")
print(inventory)

inventory = client.get_stations(network="II", station="PFO",
                                level="channel")
print(inventory)

# For waveform requests that include instrument correction, the appropriate instrument response information also has to be downloaded.

# +
t = obspy.UTCDateTime("2014-08-24T10:20:44.0")  # South Napa earthquake
st = client.get_waveforms("II", "PFO", "00", "LHZ",
                          t - 10 * 60, t + 20 * 60)
inv = client.get_stations(network="II", station="PFO", location="00", channel="LHZ",
                          level="response", starttime=t - 10, endtime=t + 10)
st.plot()

st.remove_response(inventory=inv)
st.plot()
# -

# All data requested using the FDSN client can be directly saved to file using the **`filename="..."`** option. The data is then stored exactly as it is served by the data center, i.e. without first parsing by ObsPy and outputting by ObsPy.

client.get_events(starttime=t-100, endtime=t+24*3600, minmagnitude=5,
                  filename="/tmp/requested_events.xml")
client.get_stations(network="II", station="PFO", level="station",
                    filename="/tmp/requested_stations.xml")
client.get_waveforms("II", "PFO", "00", "LHZ", t - 10 * 60, t + 20 * 60,
                     filename="/tmp/requested_waveforms.mseed")
# !ls -lrt /tmp/requested*

# #### FDSN Client Exercise
#
# Use the FDSN client to assemble a small waveform dataset for on event.
#
# - search for a large earthquake (e.g. by depth or in a region of your choice, use option **`limit=5`** to keep network traffic down)

# +
from obspy.clients.fdsn import Client

client = Client("IRIS")
catalog = client.get_events(minmagnitude=8, limit=5)
print(catalog)
catalog.plot()
event = catalog[0]
print(event)
# -

# - search for stations to look at waveforms for the event. stations should..
#     * be available at the time of the event
#     * use a vertical 1 Hz stream ("LHZ", to not overpower our network..)
#     * be in a narrow angular distance around the event (e.g. 90-91 degrees)
#     * adjust your search so that only a small number of stations (e.g. 3-6) match your search criteria
#     * Once you found a good set of stations, please use `level="response"` as you will need the response later on.

# +
origin = event.origins[0]
t = origin.time

inventory = client.get_stations(longitude=origin.longitude, latitude=origin.latitude,
                                minradius=101, maxradius=103,
                                starttime=t, endtime =t+100,
                                channel="LHZ", level="response",
                                matchtimeseries=True)
print(inventory)
# -

# - for each of these stations download data of the event, e.g. a couple of minutes before to half an hour after the event
# - put all data together in one stream (put the `get_waveforms()` call in a try/except/pass block to silently skip stations that actually have no data available)
# - print stream info, plot the raw data

# +
from obspy import Stream
st = Stream()

for network in inventory:
    for station in network:
        try:
            st += client.get_waveforms(network.code, station.code, "*", "LHZ",
                                       t - 5 * 60, t + 30 * 60)
        except:
            pass

print(st)
st.plot()
# -

# - correct the instrument response for all stations and plot the corrected data

st.remove_response(inventory=inventory, water_level=20)
st.plot()

# If you have time, assemble and plot another similar dataset (e.g. like before stations at a certain distance from a big event, or use Transportable Array data for a big event, etc.)

# #### Routed/Federated Clients
#
# Finally, all of the above queries can not only be directed at one specific data center, but instead can be sent to either ..
#
#  * [IRIS Federator](https://service.iris.edu/irisws/fedcatalog/1/), or
#  * [EIDA routing](http://www.orfeus-eu.org/data/eida/webservices/routing/) service
#
# .. to query data across all existing FDSN web service data centers (or at least those registered with IRIS/EIDA).
# These federation/routing services support station lookup and waveform request queries.
#
# This example queries the IRIS Federator which in turn combines data from all other FDSN web service data centers.

# +
from obspy.clients.fdsn import RoutingClient

client = RoutingClient('iris-federator')  # or use 'eida-routing' for the EIDA routing service

# Tohoku Earthquake
t = obspy.UTCDateTime('2011-03-11T14:46:24+09')

st = client.get_waveforms(
    channel="LHZ", starttime=t, endtime=t+20,
    latitude=50, longitude=14, maxradius=10)
print(st)
