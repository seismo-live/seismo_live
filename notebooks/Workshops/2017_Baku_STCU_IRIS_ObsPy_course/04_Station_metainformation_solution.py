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
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.7)">Handling Station Metadata</div>
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

# - for station metadata, the de-facto standard of the future (replacing SEED/RESP) is [FDSN StationXML](http://www.fdsn.org/xml/station/)
# - FDSN StationXML files can be read using **`read_inventory()`**

# +
import obspy

inventory = obspy.read_inventory("./data/station_PFO.xml")
print(type(inventory))
# -

# - the nested ObsPy Inventory class structure (Inventory/Station/Channel/Response/...) is closely modelled after FDSN StationXML
# <img src="images/Inventory.svg" width=90%>

# !head data/station_BFO.xml

print(inventory)

network = inventory[0]
print(network)

station = network[0]
print(station)

channel = station[0]
print(channel)

print(channel.response)

st = obspy.read("./data/waveform_PFO.mseed")
print(st)

inv = obspy.read_inventory("./data/station_PFO.xml", format="STATIONXML")

print(st[0].stats)

# - the instrument response can be deconvolved from the waveform data using the convenience method **`Stream.remove_response()`**
# - evalresp is used internally to calculate the instrument response

st.plot()
st.remove_response(inventory=inv)
st.plot()

# - several options can be used to specify details of the deconvolution (water level, frequency domain prefiltering), output units (velocity/displacement/acceleration), demeaning, tapering and to specify if any response stages should be omitted

st = obspy.read("./data/waveform_PFO.mseed")
st.remove_response(inventory=inv, water_level=60, pre_filt=(0.01, 0.02, 8, 10), output="DISP")
st.plot()

# Finally, if station metadata is not available from the operator or from the data center serving the data, or in case of temporary station deployments and field campaigns, full station metadata can be assembled using ObsPy, including response information gathered from the [IRIS Nominal Response Library (NRL)](http://ds.iris.edu/NRL/):
# <img src="images/nrl.png" width=70%>

# +
import obspy
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from obspy.clients.nrl.client import NRL


# We'll first create all the various objects. These strongly follow the
# hierarchy of StationXML files.
inv = Inventory(
    # We'll add networks later.
    networks=[],
    # The source should be the id whoever create the file.
    source="ObsPy-Tutorial")

net = Network(
    # This is the network code according to the SEED standard.
    code="XX",
    # A list of stations. We'll add one later.
    stations=[],
    description="A test stations.",
    # Start-and end dates are optional.
    start_date=obspy.UTCDateTime(2016, 1, 2))

sta = Station(
    # This is the station code according to the SEED standard.
    code="ABC",
    latitude=1.0,
    longitude=2.0,
    elevation=345.0,
    creation_date=obspy.UTCDateTime(2016, 1, 2),
    site=Site(name="First station"))

cha = Channel(
    # This is the channel code according to the SEED standard.
    code="HHZ",
    # This is the location code according to the SEED standard.
    location_code="",
    # Note that these coordinates can differ from the station coordinates.
    latitude=1.0,
    longitude=2.0,
    elevation=345.0,
    depth=10.0,
    azimuth=0.0,
    dip=-90.0,
    sample_rate=200)

# By default this accesses the always up-to-date NRL online.
# Offline copies of the NRL can also be used instead.
nrl = NRL()
# The contents of the NRL can be explored interactively in a Python prompt,
# see API documentation of NRL submodule:
# http://docs.obspy.org/packages/obspy.clients.nrl.html
# Here we assume that the end point of data logger and sensor are already
# known:
response = nrl.get_response(
    sensor_keys=['Nanometrics', 'Trillium Compact 120 (Vault, Posthole, OBS)', '1500 V/m/s'],
    datalogger_keys=['REF TEK', 'RT 130 & 130-SMA', '1', '200'])


# Now tie it all together.
cha.response = response
sta.channels.append(cha)
net.stations.append(sta)
inv.networks.append(net)

# And finally write it to a StationXML file. We also force a validation against
# the StationXML schema to ensure it produces a valid StationXML file.
#
# Note that it is also possible to serialize to any of the other inventory
# output formats ObsPy supports.
print(inv)
print(inv[0][0][0])
inv.write("station.xml", format="stationxml", validate=True)

# +
cha = inv[0][0][0]

cha.plot(min_freq=0.001)
print(cha)
# -

# The contents of the NRL can be inspected in an interactive shell:

nrl.sensors

nrl.sensors['Streckeisen']

nrl.dataloggers
