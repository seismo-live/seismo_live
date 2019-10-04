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
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)">Handling Event Metadata</div>
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

# - for event metadata, the de-facto standard is [QuakeML (an xml document structure)](https://quake.ethz.ch/quakeml/)
# - QuakeML files can be read using **`read_events()`**

# +
from obspy import read_events

catalog = read_events("./data/event_tohoku_with_big_aftershocks.xml")
print(catalog)
# -

# - **`read_events()`** function returns a **`Catalog`** object, which is
# a collection of **`Event`** objects.

print(type(catalog))
print(type(catalog[0]))

event = catalog[0]
print(event)

# - Event objects are again collections of other resources.
# - the nested ObsPy Event class structure (Catalog/Event/Origin/Magnitude/FocalMechanism/...) is closely modelled after QuakeML
# <img src="images/Event.svg" width=90%>

print(type(event.origins))
print(type(event.origins[0]))
print(event.origins[0])

print(type(event.magnitudes))
print(type(event.magnitudes[0]))
print(event.magnitudes[0])

# +
# try event.<Tab> to get an idea what "children" elements event has
# -

# - The Catalog object contains some convenience methods to make
# working with events easier.
# - for example, the included events can be filtered with various keys.

largest_magnitude_events = catalog.filter("magnitude >= 7.8")
print(largest_magnitude_events)

# - There is a basic preview plot using the matplotlib basemap module.

catalog.plot(projection="local");

# - a (modified) Catalog can be output to file (currently there is write support for QuakeML only)

largest_magnitude_events.write("/tmp/large_events.xml", format="QUAKEML")
# !ls -l /tmp/large_events.xml

# - the event type classes can be used to build up Events/Catalogs/Picks/.. from scratch in custom processing work flows and to share them with other researchers in the de facto standard format QuakeML

# +
from obspy import UTCDateTime
from obspy.core.event import Catalog, Event, Origin, Magnitude
from obspy.geodetics import FlinnEngdahl

# cat = Catalog()
cat.description = "Just a fictitious toy example catalog built from scratch"

e = Event()
e.event_type = "not existing"

o = Origin()
o.time = UTCDateTime(2014, 2, 23, 18, 0, 0)
o.latitude = 47.6
o.longitude = 12.0
o.depth = 10000
o.depth_type = "operator assigned"
o.evaluation_mode = "manual"
o.evaluation_status = "preliminary"
o.region = FlinnEngdahl().get_region(o.longitude, o.latitude)

m = Magnitude()
m.mag = 7.2
m.magnitude_type = "Mw"

m2 = Magnitude()
m2.mag = 7.4
m2.magnitude_type = "Ms"

# also included could be: custom picks, amplitude measurements, station magnitudes,
# focal mechanisms, moment tensors, ...

# make associations, put everything together
cat.append(e)
e.origins = [o]
e.magnitudes = [m, m2]
m.origin_id = o.resource_id
m2.origin_id = o.resource_id

print(cat)
cat.write("/tmp/my_custom_events.xml", format="QUAKEML")
# !cat /tmp/my_custom_events.xml
