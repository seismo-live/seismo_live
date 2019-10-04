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
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)">Handling Station Metadata</div>
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

# - for station metadata, the de-facto standard of the future (replacing SEED/RESP) is [FDSN StationXML](http://www.fdsn.org/xml/station/)
# - FDSN StationXML files can be read using **`read_inventory()`**

from obspy import read_inventory
# real-world StationXML files often deviate from the official schema definition
# therefore file-format autodiscovery sometimes fails and we have to force the file format
inventory = read_inventory("./data/station_PFO.xml", format="STATIONXML")
print(type(inventory))

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

from obspy import read
st = read("./data/waveform_PFO.mseed")
print(st)

inv = read_inventory("./data/station_PFO.xml", format="STATIONXML")

print(st[0].stats)

# - the instrument response can be deconvolved from the waveform data using the convenience method **`Stream.remove_response()`**
# - evalresp is used internally to calculate the instrument response

st.plot()
st.remove_response(inventory=inv)
st.plot()

# - several options can be used to specify details of the deconvolution (water level, frequency domain prefiltering), output units (velocity/displacement/acceleration), demeaning, tapering and to specify if any response stages should be omitted

st = read("./data/waveform_PFO.mseed")
st.remove_response(inventory=inv, water_level=60, pre_filt=(0.01, 0.02, 8, 10), output="DISP")
st.plot()

# - station metadata not present in StationXML yet but in Dataless SEED or RESP files can be used for instrument correction using the `.simulate()` method of Stream/Trace in a similar fashion
