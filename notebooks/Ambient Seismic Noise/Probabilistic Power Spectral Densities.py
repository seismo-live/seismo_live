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
#             <div style="font-size: xx-large ; font-weight: 900 ; color: rgba(0 , 0 , 0 , 0.8) ; line-height: 100%">Noise</div>
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)">Lab: Probabilistic Power Spectral Densities</div>
#         </div>
#     </div>
# </div>
#
#
# Seismo-Live: http://seismo-live.org
#
# ##### Authors:
# * Tobias Megies ([@megies](https://github.com/megies))
#
# ---

# %matplotlib inline
import matplotlib.pyplot as plt
import warnings
plt.style.use("bmh")
plt.rcParams['figure.figsize'] = 10, 6
warnings.filterwarnings("ignore")

#  * read waveform data from file `data/GR.FUR..BHN.D.2015.361` (station `FUR`, [LMU geophysical observatory in Fürstenfeldbruck](https://www.geophysik.uni-muenchen.de/observatory/seismology))
#  * read corresponding station metadata from file `data/station_FUR.stationxml`
#  * print info on both waveforms and station metadata

# +
from obspy import read, read_inventory

st = read("data/GR.FUR..BHN.D.2015.361")
inv = read_inventory("data/station_FUR.stationxml")

print(st)
print(inv)
inv.plot(projection="ortho");
# -

#  * compute probabilistic power spectral densities using `PPSD` class from obspy.signal, see http://docs.obspy.org/tutorial/code_snippets/probabilistic_power_spectral_density.html (but use the inventory you read from StationXML as metadata)
#  * plot the processed `PPSD` (`plot()` method attached to `PPSD` object)

# +
from obspy.signal import PPSD

tr = st[0]
ppsd = PPSD(stats=tr.stats, metadata=inv)

ppsd.add(tr)
ppsd.plot()
# -

# Since longer term stacks would need too much waveform data and take way too long to compute, we prepared one year continuous data preprocessed for a single channel of station `FUR` to play with..
#
#  * load long term pre-computed PPSD from file `PPSD_FUR_HHN.npz` using `PPSD`'s `load_npz()` staticmethod (i.e. it is called directly from the class, not an instance object of the class)
#  * plot the PPSD (default is full time-range, depending on how much data and spread is in the data, adjust `max_percentage` option of `plot()` option)  (might take a couple of minutes..!)
#  * do a cumulative plot (which is good to judge non-exceedance percentage dB thresholds)

# +
from obspy.signal import PPSD

ppsd = PPSD.load_npz("data/PPSD_FUR_HHN.npz")
# -

ppsd.plot(max_percentage=10)
ppsd.plot(cumulative=True)

#  * do different stacks of the data using the [`calculate_histogram()` (see docs!)](http://docs.obspy.org/packages/autogen/obspy.signal.spectral_estimation.PPSD.calculate_histogram.html) method of `PPSD` and visualize them
#  * compare differences in different frequency bands qualitatively (anthropogenic vs. "natural" noise)..
#    * nighttime stack, daytime stack
#  * advanced exercise: Use the `callback` option and use some crazy custom callback function in `calculate_histogram()`, e.g. stack together all data from birthdays in your family.. or all German holidays + Sundays in the time span.. or from dates of some bands' concerts on a tour.. etc.

ppsd.calculate_histogram(time_of_weekday=[(-1, 0, 2), (-1, 22, 24)])
ppsd.plot(max_percentage=10)
ppsd.calculate_histogram(time_of_weekday=[(-1, 8, 16)])
ppsd.plot(max_percentage=10)

#  * do different stacks of the data using the [`calculate_histogram()` (see docs!)](http://docs.obspy.org/packages/autogen/obspy.signal.spectral_estimation.PPSD.calculate_histogram.html) method of `PPSD` and visualize them
#  * compare differences in different frequency bands qualitatively (anthropogenic vs. "natural" noise)..
#    * weekdays stack, weekend stack

ppsd.calculate_histogram(time_of_weekday=[(1, 0, 24), (2, 0, 24), (3, 0, 24), (4, 0, 24), (5, 0, 24)])
ppsd.plot(max_percentage=10)
ppsd.calculate_histogram(time_of_weekday=[(6, 0, 24), (7, 0, 24)])
ppsd.plot(max_percentage=10)

#  * do different stacks of the data using the [`calculate_histogram()` (see docs!)](http://docs.obspy.org/packages/autogen/obspy.signal.spectral_estimation.PPSD.calculate_histogram.html) method of `PPSD` and visualize them
#  * compare differences in different frequency bands qualitatively (anthropogenic vs. "natural" noise)..
#    * seasonal stacks (e.g. northern hemisphere autumn vs. spring/summer, ...)

ppsd.calculate_histogram(month=[10, 11, 12, 1])
ppsd.plot(max_percentage=10)
ppsd.calculate_histogram(month=[4, 5, 6, 7])
ppsd.plot(max_percentage=10)

#  * do different stacks of the data using the [`calculate_histogram()` (see docs!)](http://docs.obspy.org/packages/autogen/obspy.signal.spectral_estimation.PPSD.calculate_histogram.html) method of `PPSD` and visualize them
#  * compare differences in different frequency bands qualitatively (anthropogenic vs. "natural" noise)..
#    * stacks by specific month
#    * maybe even combine several of above restrictions.. (e.g. only nighttime on weekends)


