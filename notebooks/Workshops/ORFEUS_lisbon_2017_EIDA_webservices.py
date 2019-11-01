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
#             <div style="font-size: xx-large ; font-weight: 900 ; color: rgba(0 , 0 , 0 , 0.8) ; line-height: 100%">ORFEUS Workshop - Lisbon 2017</div>
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)">ORFEUS EIDA Webservices</div>
#         </div>
#     </div>
# </div>

# Seismo-Live: http://seismo-live.org
#
# ##### Authors:
# * Mathijs Koymans ([@jollyfant](https://github.com/jollyfant))
#
# ---

# ## 1 Basic Webservice Usage
# ## 1.1 Introduction
# EIDA webservices are designed to provide programmatic access to waveform data and instrument metadata from EIDA. FDSN standerdised webservices are running since 2015 and are scheduled to replace Arclink and other deprecated procotols in the near future. Because webservices requests are URLs It is possible to communicate directly with the webservice APIs in a browser, command-line tools (e.g. curl; wget) or through abstracted clients (e.g. [ObsPy](http://obspy.org), [fdsnws-fetch](https://github.com/andres-h/fdsnws_scripts/blob/master/fdsnws_fetch.py)).
#
# Webservices are identified by the service domain (URL) that is data center specific, a label that identifies the service (e.g. dataselect; station) and a list of request options (e.g. stream identifiers or time window) included in its query string. In this initial exercise we will introduce five webservices:
#
# * 1.2 FDSNWS-Dataselect - Raw waveform service
# * 1.3 FDSNWS-Station - Station metadata and instrument specifics
# * 1.4 EIDAWS-Routing - Service routing within EIDA
# * 1.5 EIDAWS-WFCatalog - Waveform metadata
# * 1.6 EIDA Mediator - Automatically federated requests across EIDA
#
# In this notebook we will practise direct communication with the webservice APIs in addition to recommended and more convenient workflows using ObsPy.

# ## 1.2 FDSNWS-Dataselect
# ### 1.2.1 Interacting with the API
# The following example makes a request to the FDSNWS-Dataselect API hosted at ORFEUS Data Center (http://orfeus-eu.org). We will request a 10-minute window of miniSEED data from a single station. The data will be read and plotted using ObsPy. Alternatively, we could save the data to disk. The service label for FDSNWS-Dataselect is:
#
# > fdsnws/dataselect/1/query

# +
# %matplotlib inline

# Import the read module from ObsPy
from obspy import read

# The URL that points to the dataselect service
# The label that identifies the service
SERVICE_DOMAIN = "http://www.orfeus-eu.org"
LABEL = "fdsnws/dataselect/1/query"

# The 10-minute time window tuple
starttime, endtime = ("2016-01-01T00:00:00", "2016-01-01T00:10:00")

# Get the SEED codes, we will use wildcards for location, channel
network, station, location, channel = "NL", "HGN", "*", "*"

# Create a query string
queryString = "&".join([
    "network=%s" % network,
    "station=%s" % station,
    "location=%s" % location,
    "channel=%s" % channel,
    "starttime=%s" % starttime,
    "endtime=%s" % endtime
])

# The URL that we are requesting data from
# Try visiting this URL in your browser:
# http://www.orfeus-eu.org/fdsnws/dataselect/1/query?network=NL&station=HGN&location=*&channel=*&starttime=2016-01-01T00:00:00&endtime=2016-01-01T00:10:00
st = read("%s/%s?%s" % (SERVICE_DOMAIN, LABEL, queryString))

# Plot the data returned by the webservice
st.plot();
# -

# ### 1.2.2 Waveforms through ObsPy (recommended usage)
# Alternatively we can use the ObsPy library to communicate with the API through an abstracted client. All we need to do is call an ObsPy function with our time window constraint and SEED identifiers. This function will do all the work of the previous exercise for us internally and make the result available for use within ObsPy.
#
# **Note:** Instead of building the URL yourself in the previous exercise, when working with ObsPy it is recommended that the client class is used.

# +
# Include the Client class from ObsPy
from obspy.clients.fdsn import Client

# Create an ObsPy Client that points to ODC (http://www.orfeus-eu.org)
client = Client("ODC")

# Get the waveforms for the same trace identifiers and time window
st = client.get_waveforms(network, station, location, channel, starttime, endtime)

# Plot identical result
st.plot();
# -

# ## 1.3 FDSNWS-Station
# ### 1.3.1 Interacting with the API
# The fdsnws-station service works similar to the fdsnws-dataselect but has a service different label (*station* instead of *dataselect*). The response of this webservice is StationXML by default. In the following example we will however request the output formatted as text for clarity. The label for this webservice is:
#
# > fdsnws/station/1/query

# +
# Import a library to make a HTTP request to the webservice
import requests

# The URL that points to the station service
SERVICE_DOMAIN = "http://www.orfeus-eu.org"
LABEL = "fdsnws/station/1/query"

# Get the SEED codes for the entire NL network
network, station, location, channel = "NL", "*", "*", "*"

# The query string includes our seed identifiers
# and we request output format text
queryString = "&".join([
    "network=%s" % network,
    "station=%s" % station,
    "location=%s" % location,
    "channel=%s" % channel,
    "format=text",
    "level=station"
])

# The URL that we are requesing
# Try this in your browser:
# http://www.orfeus-eu.org/fdsnws/station/1/query?network=NL&station=*&location=*&channel=*&format=text
r = requests.get("%s/%s?%s" % (SERVICE_DOMAIN, LABEL, queryString))

# This will print station information for all stations in network NL
print(r.text)
# -

# Practically, the data would be requested in StatonXML format and saved to file, to be further used during data processing. In the following exercise we will read the data directly into ObsPy. Note again that when working with ObsPy, using the client class is the best solution.

# ### 1.3.2 Station Metadata through ObsPy (recommended usage)
# Alternatively, we use an ObsPy client to be able to directly manipulate the data in ObsPy. In the following example we request the instrument response for a single channel and print the response information. In combination with the raw waveform data returned from dataselect service we can deconvolve the frequency response for this sensor.

# +
# We will request instrument metadata for a single trace
network, station, location, channel = "NL", "HGN", "02", "BH*"

# We pass level=response to request instrument response metadata
inv = client.get_stations(
    network=network,
    station=station,
    location=location,
    channel=channel,
    level="response"
)

# This object now has response information for the selected trace (NL.HGN.02.BHZ)
for network in inv:
    for station in network:
        for channel in station:
            print(channel.response)

# Deconvolve instrument response
st.remove_response(inventory=inv)

# Plot the data (output units = velocity)
st.plot();
# -

# ## 1.4 EIDAWS-Routing
# The seismic archive of EIDA is distributed across 11 different data centers, called EIDA Nodes. EIDAWS-routing helps you to find data within this federated data archive. If you don't know which EIDA node holds your data of interest the routing service will provide you with the appropriate EIDA node and corresponding webservice URL to be queried.
#
# In this example we will request the "get" format (i.e. URLs that hold the data) for four networks. We are asking for all routes to the station webservice. The label for this service is:
#
# > eidaws/routing/1/query
#
# **Note:** routing and communication with all EIDA nodes individually can be omitted by using the EIDA Mediator in federated mode (see section 1.6).

# +
# The URL that points to the routing service (notice the different eidaws label)
SERVICE_DOMAIN = "http://www.orfeus-eu.org"
LABEL = "eidaws/routing/1/query"

# Network codes must be comma delimited
network = ",".join(["HL", "GE", "NL", "KO"])

# The query string includes our network codes and our output format must is set as URLs (get)
# We specify the service as fdsnws-station (change this to dataselect)
queryString = "&".join([
    "network=%s" % network,
    "format=get",
    "service=station"
])

# The URL that we are requesing
# Try this in your browser:
# http://www.orfeus-eu.org/eidaws/routing/1/query?network=HL,GE,NL,KO&format=get
r = requests.get("%s/%s?%s" % (SERVICE_DOMAIN, LABEL, queryString))

# Should print four routes to different data centers
# Here we can find station metadata for these four networks respectively
# We make a request to all returned routes (status 200 indicates success!)
for line in r.text.split("\n"):
    r = requests.get(line)
    print("[%i] %s" % (r.status_code, line))
# -

# ## 1.5 EIDAWS-WFCatalog
# The WFCatalog is a catalogue of seismic waveform metadata. This is not to be confused with station metadata but contains purely metadata describing the waveforms. These metadata include availability information (e.g. gaps), sample metrics (e.g. mean, standard deviations, median values) and miniSEED header flags.
#
# The EIDAWS-WFCatalog webservice returns quality metrics from raw waveform data. The WFCatalog can serve as a powerful waveform index for data discovery by appending filters (e.g. lt, ge) to the query string. This can help identify waveforms with metric values below or above a certain threshold. The label for this service is:
#
# > eidaws/wfcatalog/1/query

# +
# The URL that points to the routing service (notice the different eidaws label)
SERVICE_DOMAIN = "http://www.orfeus-eu.org"
LABEL = "eidaws/wfcatalog/1/query"

# The start and end date for the metrics
# Feel free to change the window
starttime, endtime = ("2010-11-01", "2010-11-07")

# Network codes must be comma delimited
network, station, location, channel = "NL.HGN.02.BHZ".split(".")

# The query string includes our seed identifiers, temporal constraints, we ask for sample metrics to be included
# include can be either (default, sample, header, all)
# We request metrics for daily waveforms with an availability over 50%
# Try changing the percent_availability to 100 - less documents will be returned
queryString = "&".join([
    "network=%s" % network,
    "station=%s" % station,
    "location=%s" % location,
    "channel=%s" % channel,
    "starttime=%s" % starttime,
    "endtime=%s" % endtime,
    "include=sample",
    "percent_availability_ge=50"
])

# Try this in your browser:
# http://www.orfeus-eu.org/eidaws/wfcatalog/1/query?network=NL&station=HGN&location=02&channel=BHZ&start=2010-11-01&end=2010-11-07&include=sample
r = requests.get("%s/%s?%s" % (SERVICE_DOMAIN, LABEL, queryString))

# Should print JSON response of quality metrics for three days.
r.json()
# -

# ## 1.6 EIDA Mediator
# The EIDA mediator (beta) can automatically route and retrieve requests federated between EIDA nodes. This prevents using from having to query the routing service before making data requests. There is a single entry poiny to the entire archive available within EIDA as demonstrated below. Currently there is supported for federated mode between **station** and **dataselect**. Federation of **WFCatalog** requests will be supported in the future.

# +
# The URL that points to the routing service (the EIDA mediator is hosted by ETHZ)
SERVICE_DOMAIN = "http://mediator-devel.ethz.ch"
LABEL = "fdsnws/station/1/query"

# Network codes must be comma delimited
# Networks are federated across 4 different EIDA nodes
network = ",".join(["HL", "GE", "NL", "KO"])

# Creathe queyr string and append all networks
# We ask for level=network to limit the amount of data returned for clarity
queryString = "&".join([
    "network=%s" % network,
    "level=network"
])

# Try this in your browser:
# http://mediator-devel.ethz.ch/fdsnws/station/1/query?network=HL,GE,NL&level=network
##### This currently does not seem to work.
# r = requests.get("%s/%s?%s" %(SERVICE_DOMAIN, LABEL, queryString))

# StationXML for all four networks
# print(r.text)
# -

# ## Graphical user interfaces
# The following tools are available on orfeus-eu.org and are built on top of the discussed webservices. Please note that these interfaces currently only work for data archived at ORFEUS Data Center.
#
# > http://www.orfeus-eu.org/data/odc/quality

# # 2 Advanced Example - Webservices pipeline
# ## 2.1 Introduction
# This example demonstrates the use of FDSN webservices in a processing pipeline. The goal of this exercise is to download raw waveform data from stations surrounding an earthquake. This pipeline is based on functionality provided with ObsPy.

# +
# Define the module imports
import requests
import math

from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
from obspy import read, UTCDateTime

import datetime
import dateutil.parser


# -

# ## 2.2 FDSNWS-Event
#
# We define a function that collects event information from fdsnws-event. We pass an event identifier to the webservice, parse the response and return an Event class that has **location**, **origin time**, and **depth** attributes. The event data is requested from the seismicportal webservice provided by the EMSC.

def getEvent(identifier):

    # Try in your browser:
    # http://www.seismicportal.eu/fdsnws/event/1/query?eventid=20170720_0000091&format=text
    # Service address
    FDSN_EVENT = "http://www.seismicportal.eu/fdsnws/event/1/query"

    # Define class for Events
    class Event():
        def __init__(self, line):
            self.id, self.time, self.latitude, self.longitude, self.depth = line.split("|")[:5]
            self.latitude = float(self.latitude)
            self.longitude = float(self.longitude)
            self.depth = float(self.depth)
            
    # We query for a single event identifier and request a text format return
    queryString = "&".join([
      "eventid=%s" % identifier,
      "format=text"
    ])

    # Create the query for an event identifier
    r = requests.get("%s?%s" % (FDSN_EVENT, queryString))

    # Split by lines and remove head & tail
    lines = r.text.split("\n")[1:-1]

    # Return Event classes for each entry
    return list(map(Event, lines))[0]


# Should print a single Event instance
print(getEvent("20170720_0000091"))


# ## 2.3 FDSNWS-Station
#
# Define a function that can find the stations around an event. We pass the Event instance to the function and call the station webservice to return stations within 20 degrees arc-distance of this event location. We parse the response and return a map of station instances with attributes network, station, and location.

def getStations(event):

    # Try it in your browser:
    # http://orfeus-eu.org/fdsnws/station/1/query?latitude=30&longitude=30&maxradius=20&format=text
    # Service address
    FDSN_STATION = "http://orfeus-eu.org/fdsnws/station/1/query"
    MAX_RADIUS = 20

    # Define a Station class
    class Station():
        def __init__(self, line):
            self.network, self.station, self.latitude, self.longitude = line.split("|")[:4]
            self.latitude = float(self.latitude)
            self.longitude = float(self.longitude)

    # We query with the event location and a maximum radius around the event
    queryString = "&".join([
      "latitude=%s" % event.latitude,
      "longitude=%s" % event.longitude,
      "maxradius=%s" % MAX_RADIUS,
      "format=text"
    ])

    # Request from webservice
    r = requests.get("%s?%s" % (FDSN_STATION, queryString))

    # Split by lines and remove head & tail
    lines = r.text.split("\n")[1:-1]

    # Return Event classes for each entry
    return map(Station, lines)


# Should print a map (array) of Station instances
print(getStations(getEvent("20170720_0000091")))

# ## 2.4 Theoretical Arrival Times
#
# Define a function that calculates the theoretical P arrival time at a station location using the TauP module in ObsPy. The function takes an Event and Station instance. The arc-distance in degrees between the source and receiver is calculated using the *haversine function* (see below).

# +
# We use the iasp91 reference model
TAUP_MODEL = TauPyModel(model="iasp91")

def getPArrival(event, station):

    # Determine the arc distance using the haversine formula
    arcDistanceDegrees = locations2degrees(
      event.latitude,
      station.latitude,
      event.longitude,
      station.longitude
    )

    # Calculate the theoretical P-arrival time
    arrivals = TAUP_MODEL.get_travel_times(
      source_depth_in_km=1E-3 * event.depth,
      distance_in_degree=arcDistanceDegrees,
      phase_list=["P"]
    )

    # Add the theorical P-arrival delta to the event time
    return UTCDateTime(event.time) + arrivals[0].time


# -

# Definition of the havesine function, we pass two latitudes and longitudes and return the arc-distance in degrees. This is a supplementary function.

# ## 2.5 FDSNWS-Dataselect
#
# The main body of the script that collects an event with event identifier 20170720_0000091. We loop over all the stations returned by the getStations function within 20 degrees arc-distance of the event. In each iteration, we make a call to fdsnws-dataselect to collect the waveform data for all stations between 300 seconds before, and 1200 seconds after the theoretical P-arrival time.
#
# This data (channel BH?) is loaded in to ObsPy using the read function, filtered and plotted. After the first iteration the loop is broken. Alternatively, all data can be saved to disk.
#

# +
FDSN_DATASELECT = "http://orfeus-eu.org/fdsnws/dataselect/1/query"
EVENT_IDENTIFIER = "20170720_0000091"

# Get the event
event = getEvent(EVENT_IDENTIFIER)

# Go over all stations returned in the radius
for station in getStations(event):

    # Get the theoretical (TauP) pArrval from event to station
    stationArrivalTime = getPArrival(event, station)

    # Create the query for fdsn-dataselect
    # between 300 seconds before & 1200 seconds after the theoretical P-arrival
    queryString = "&".join([
        "network=%s" % station.network,
        "station=%s" % station.station,
        "starttime=%s" % (stationArrivalTime - 300).isoformat(),
        "endtime=%s" % (stationArrivalTime + 1200).isoformat(),
        "channel=BH?"
    ])

    # Get the waveform data and read to ObsPy Stream
    # Empty responses are skipped 
    try:
        st = read("%s?%s" % (FDSN_DATASELECT, queryString))
    except Exception:
        continue

    # Use with ObsPy and apply a filter, then plot
    # Alternatively, we would save the data to a file
    st.filter("lowpass", freq=0.5)
    st.plot()

    # Break after the first result
    break
# -

# ## Acknowledgements
# Thanks to Lion Krischer and Seismo-Live for hosting this notebook.
