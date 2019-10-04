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
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.7)">Handling Time</div>
#         </div>
#     </div>
# </div>
# image: User:Abbaszade656 / Wikimedia Commons / <a href="http://creativecommons.org/licenses/by-sa/4.0/">CC-BY-SA-4.0</a>

# ## Workshop for the "Training in Network Management Systems and Analytical Tools for Seismic"
# ### Baku, October 2018
#
#
# Seismo-Live: http://seismo-live.org
#
# ##### Authors:
# * Lion Krischer ([@krischer](https://github.com/krischer))
# * Tobias Megies ([@megies](https://github.com/megies))
#
# ---

# ![](images/obspy_logo_full_524x179px.png)

# This is a bit dry but not very difficult and important to know. It is used **everywhere** in ObsPy!
#
#
# * All absolute time values are consistently handled with this class.
# * Based on a nanosecond precision POSIX integer timestamp for accuracy.
# * Timezone can be specified at initialization (if necessary).
# * In Coordinated Universal Time (UTC) so no need to deal with timezones, daylight savings, ...
#
# ---

# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 12, 8

# ---
#
# ## Features of **`UTCDateTime`**
#
# #### Initialization

# +
from obspy import UTCDateTime

print(UTCDateTime("2014-08-24T10:20:44.0"))        # mostly time strings defined by ISO standard
print(UTCDateTime("2014-08-24T01:20:44.0-09:00"))  # non-UTC timezone input
print(UTCDateTime(2014, 8, 24, 10, 20, 44))        # year, month, day, hour, min, sec, musec
print(UTCDateTime(1408875644.0))                   # timestamp
# -

# Current time can be initialized by leaving out any arguments
print(UTCDateTime())

# #### Attribute Access

time = UTCDateTime("2014-08-24T10:20:44.0")
print(time.year)
print(time.julday)
print(time.timestamp)
print(time.weekday)
# try time.<Tab>

# #### Handling time differences
#
# * "**`+`**/**`-`**" defined to add seconds to an **`UTCDateTime`** object
# * "**`-`**" defined to get time difference of two **`UTCDateTime`** objects

time = UTCDateTime("2014-08-24T10:20:44.0")
print(time)

# one hour later
print(time + 3600)

# Time differences
time2 = UTCDateTime(2015, 1, 1)
print(time2 - time)

# ### Exercises
#
# Calculate the number of days passed since the 2014 South Napa earthquake (the timestamp used above).

print((UTCDateTime() - UTCDateTime("2014-08-24T11:20:44.000000Z")) / 86400)

# Make a list of 10 UTCDateTime objects, starting today at 10:00 with a spacing of 90 minutes.

# +
t = UTCDateTime(2017, 9, 18, 10)

times = []
for i in range(10):
    t2 = t + i * 90 * 60
    times.append(t2)

times
# -

# Below is a list of strings with origin times of magnitude 8+ earthquakes since 2000 (fetched from IRIS). Assemble a list of interevent times in days. Use matplotlib to display a histogram.

times = ["2000-11-16T04:54:56",
         "2001-06-23T20:33:09",
         "2003-09-25T19:50:07",
         "2004-12-23T14:59:00",
         "2004-12-26T00:58:52",
         "2005-03-28T16:09:35",
         "2006-05-03T15:26:39",
         "2006-06-01T18:57:02",
         "2006-06-05T00:50:31",
         "2006-11-15T11:14:14",
         "2007-01-13T04:23:23",
         "2007-04-01T20:39:56",
         "2007-08-15T23:40:58",
         "2007-09-12T11:10:26",
         "2009-09-29T17:48:11",
         "2010-02-27T06:34:13",
         "2011-03-11T05:46:23",
         "2012-04-11T08:38:37",
         "2012-04-11T10:43:10",
         "2013-05-24T05:44:49",
         "2014-04-01T23:46:47",
         "2015-09-16T22:54:32",
         "2017-09-08T04:49:21"]

# +
import matplotlib.pyplot as plt

inter_event_times = []

for i in range(1, len(times)):
    dt = UTCDateTime(times[i]) - UTCDateTime(times[i-1])
    dt = dt / (3600 * 24)
    inter_event_times.append(dt)

plt.hist(inter_event_times, bins=range(0, 1000, 100))
plt.xlabel("Magnitude 8+ interevent times since 2000 [days]")
plt.ylabel("count")
plt.show()
