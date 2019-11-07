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
# # Figure 6:  Data Quality - Detecting Wrong Times with Synthetics
#
# This notebook is part of the supplementary materials for the Syngine paper and reproduces figure 6.
#
# This notebook creates the phase relative times figure. Requires matplotlib >= 1.5 and an ObsPy version (>= 1.0) with the syngine client as well as instaseis.
#
# ##### Authors:
# * Lion Krischer ([@krischer](https://github.com/krischer))

# + {"deletable": true, "editable": true}
# %matplotlib inline

# + {"deletable": true, "editable": true}
import obspy
import numpy as np
from obspy.clients.fdsn import Client
from obspy.clients.syngine import Client as SyngineClient

import itertools

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
plt.style.use("seaborn-paper")

# + {"deletable": true, "editable": true}
c_iris = Client("IRIS")

# + {"deletable": true, "editable": true}
# Get all stations from the Caltech Regional Seismic Network activate
# at the given time.
inv = c_iris.get_stations(level="response", format="xml", channel="BHZ",
                          network="CI", location="  ",
                          startbefore=obspy.UTCDateTime(2012, 1, 1),
                          endafter=obspy.UTCDateTime(2016, 1, 1))
inv.plot(projection="local");
print(inv)

# + {"deletable": true, "editable": true}
# Downloaded from the GCMT page.
# cat = obspy.read_events("./cmtsolutions.txt", format="CMTSOLUTION").filter("latitude > -20", "longitude > 165")

# This is a prototype - we thus are very selective regarding which data we use...
training = ["smi:local/cmtsolution/201510202152A/event",
            "smi:local/cmtsolution/201212212228A/event",
            "smi:local/cmtsolution/201203090709A/event"]
validate = "smi:local/cmtsolution/201401011603A/event"
application = "smi:local/cmtsolution/201501230347A/event"

all_data = []
all_data.extend(training)
all_data.append(validate)
all_data.append(application)

cat.events = [_i for _i in cat.events if _i.resource_id.id in all_data]

print(cat)
cat.plot();

# + {"deletable": true, "editable": true}
# For all of them: download the syngine data 50 seconds before the p-phase
# and 50 seconds after for the 5s ak135f DB.
c_syngine = SyngineClient()

# + {"deletable": true, "editable": true}
# Download all syngine data.
synthetic_data = {}

# Build up bulk request.
bulk = [
    {"networkcode": "CI", "stationcode": _i.code,
     "latitude": _i.latitude, "longitude": _i.longitude} for _i in inv[0]]

# Actually download everything.
for _i, event in enumerate(cat):
    origin = [_i for _i in event.origins if _i.origin_type == "hypocenter"][0]
    mt = event.focal_mechanisms[0].moment_tensor.tensor
    print("Downloading data for event %i of %i ..." % (_i + 1, len(cat)))
    st_syn = c_syngine.get_waveforms_bulk(
        model="ak135f_5s", bulk=bulk, sourcelatitude=origin.latitude,
        sourcelongitude=origin.longitude, sourcedepthinmeters=origin.depth,
        sourcemomenttensor=[mt.m_rr, mt.m_tt, mt.m_pp, mt.m_rt, mt.m_rp, mt.m_tp],
        origintime=origin.time,
        components="Z",
        # Avoid downsampling - just upsample to the data sampling rate.
        dt=1.0 / 40.0,
        kernelwidth=10,
        units="velocity",
        starttime="P-50", endtime="P+50")

    this_event = {"event_object": event, "stream": st_syn}
    synthetic_data[event.resource_id] = this_event
    print("  -> Downloaded %i synthetic traces." % len(st_syn))

# + {"deletable": true, "editable": true}
# Reordering ... I don't want to redownload again...
data = synthetic_data
for value in data.values():
    value["synthetic_stream"] = value["stream"]
    del value["stream"]

# + {"deletable": true, "editable": true}
# Now download all real data.
for _i, value in enumerate(data.values()):
    print("Downloading data for event %i of %i ..." % (_i + 1, len(cat)))
    st_obs = c_iris.get_waveforms_bulk(
        [(tr.stats.network, tr.stats.station, "",
          "BHZ", tr.stats.starttime, tr.stats.endtime) for tr in value["synthetic_stream"]])
    value["observed_stream"] = st_obs
    print("  -> Downloaded %i observed traces." % len(st_obs))

# + {"deletable": true, "editable": true}
# Cleanup - here we make our job easy and just remove all
# stations for which we don't have data for all events.
# Real implementations will of course have to deal with this.

common_stations = set.intersection(*[set((tr.stats.network, tr.stats.station)
                                         for tr in value["observed_stream"])
                                     for value in data.values()])
print(len(common_stations))

for value in data.values():
    value["observed_stream"] = obspy.Stream(traces=[
        tr for tr in value["observed_stream"] if
        (tr.stats.network, tr.stats.station) in common_stations])
    value["synthetic_stream"] = obspy.Stream(traces=[
        tr for tr in value["synthetic_stream"] if
        (tr.stats.network, tr.stats.station) in common_stations])

# + {"deletable": true, "editable": true}
# Create a new inventory object which only contains the common stations.
from copy import deepcopy
filtered_inv = deepcopy(inv)
filtered_inv[0].stations = [_i for _i in filtered_inv[0].stations if ("CI", _i.code) in common_stations]
filtered_inv.plot(projection="local");

# + {"deletable": true, "editable": true}
from obspy.signal.cross_correlation import xcorr_pick_correction


collected_stats = []

good_data = data

# Now calculate the P phase delay for all stations.
for network, station in common_stations:
    print(network, station)
    for event, this_data in good_data.items():
        obs_tr = this_data["observed_stream"].select(network=network, station=station)[0].copy()
        syn_tr = this_data["synthetic_stream"].select(network=network, station=station)[0].copy()

        # Artificially shift the data for two stations for the "application" event.
        if event.id == application:
            if station == "BEL":
                obs_tr.stats.starttime -= 1.0
            if station == "PHL":
                obs_tr.stats.starttime += 1.0

        # p-phase should be centered in the synthetic trace.
        pick = syn_tr.stats.starttime + (syn_tr.stats.endtime - syn_tr.stats.starttime) / 2.0

        # Process data.
        obs_tr.detrend("linear")\
            .taper(max_percentage=0.05, type="hann")\
            .remove_response(inventory=filtered_inv, pre_filt=(0.02, 0.01, 1, 5))\
            .filter("bandpass", freqmin=0.08, freqmax=0.2, corners=4, zerophase=True)\
            ._ltrim(30.0)\
            ._rtrim(30.0)\
            .taper(max_percentage=0.3, type="hann")
        obs_tr.trim(obs_tr.stats.starttime - 50, obs_tr.stats.endtime + 50, pad=True, fill_value=0.0)

        # Also filter the synthetics.
        syn_tr.taper(max_percentage=0.05, type="hann")
        syn_tr.filter("bandpass", freqmin=0.08, freqmax=0.2, corners=4, zerophase=True)
        syn_tr.data = np.require(syn_tr, requirements=["C"])

        # Interpolate both to the same sample points.
        starttime = max(obs_tr.stats.starttime, syn_tr.stats.starttime)
        endtime = min(obs_tr.stats.endtime, syn_tr.stats.endtime)
        npts = int((endtime - starttime) // (1.0 / 40.0) - 1)
        obs_tr.interpolate(sampling_rate=40.0, method="lanczos", a=5,
                           starttime=starttime, npts=npts)
        syn_tr.interpolate(sampling_rate=40.0, method="lanczos", a=5,
                           starttime=starttime, npts=npts)

        print(event.id, network, station)

        # Subsample precision.
        try:
            shift, corr = xcorr_pick_correction(pick, obs_tr, pick, syn_tr, t_before=20,
                                                t_after=20, cc_maxlag=12)
        except:
            shift, corr = None, None
        collected_stats.append({"time_shift": shift, "correlation": corr,
                                "network": network, "station": station, "event": event.id})

        if "processed_observed_stream" not in this_data:
            this_data["processed_observed_stream"] = obspy.Stream()

        this_data["processed_observed_stream"] += obs_tr

        if "processed_synthetic_stream" not in this_data:
            this_data["processed_synthetic_stream"] = obspy.Stream()

        this_data["processed_synthetic_stream"] += syn_tr


# + {"deletable": true, "editable": true}
import pandas
pandas.set_option('display.max_rows', 10)
cs = pandas.DataFrame(collected_stats)
original_cs = pandas.DataFrame(collected_stats)

# + {"deletable": true, "editable": true}
# Now for all events: substract the mean time shift to account for
# unexplained structure and unknown origin time.
cs.loc[:, "time_shift"] = cs.groupby("event").time_shift.transform(lambda df: df - df.mean())
cs = cs.sort_values("station")
cs

# + {"deletable": true, "editable": true}
# Calculate the reference time shift for each station.
reference_shifts = dict(cs.loc[cs.event.isin(training)].groupby("station").time_shift.mean())
reference_shifts

# + {"deletable": true, "editable": true}
import matplotlib.pyplot as plt

def check_event(event, ax=None, ticks=False):
    _ds = cs[cs.event == event]
    mean = _ds.time_shift.mean()

    things = []

    for _i in _ds.itertuples():
        things.append({"station": _i.station,
                       "time_shift": _i.time_shift - mean,
                       "difference_to_mean": (_i.time_shift - mean) - reference_shifts[_i.station]})

    temp = pandas.DataFrame(things).sort_values("station")

    if ax is None:
        plt.figure(figsize=(15, 5))
        ax = plt.gca()
    plt.bar(np.arange(len(_ds)) - 0.4, temp.difference_to_mean, 0.8, color="0.2")
    plt.ylim(-1.4, 1.4)
    if ticks:
        plt.xticks(range(len(_ds)), temp.station, rotation="vertical")
    else:
        plt.xticks(range(len(_ds)), [""] * len(_ds), rotation="vertical")

    plt.xlim(-1, 23)
check_event(application)

# + {"deletable": true, "editable": true}
check_event(validate)

# + {"deletable": true, "editable": true}
check_event(application)


# + {"deletable": true, "editable": true}
def plot_data(event, station, show=True, legend=False):
    val = data[obspy.core.event.ResourceIdentifier(event)]
    obs = val["processed_observed_stream"].select(station=station)[0]
    syn = val["processed_synthetic_stream"].select(station=station)[0]

    plt.plot(obs.times(), obs.data * 1E6, color="0.3", linestyle="--", linewidth=5,
             label="Observed Data")
    plt.plot(syn.times(), syn.data * 1E6, color="0.1", linestyle=":", label="Synthetic Data")

    ts= original_cs[(original_cs.event == validate) &
                    (original_cs.station == "TIN")].time_shift

    plt.plot(syn.times() - float(ts), syn.data * 1E6, color="k", label="Shifted Synthetic Data")
    plt.ylabel("Velocity [$\mu$m/s]")
    plt.xlabel("Relative Time [s]")
    plt.xlim(20, 65)

    plt.xticks([30, 40, 50, 60], ["30", "40", "50", "60"])

    if legend:
        plt.legend(loc="lower left")

    plt.text(0.03, 0.92, "Station: CI.%s" % station, transform=plt.gca().transAxes, va="top")
    if show:
        plt.show()

plot_data(validate, "TIN")
plot_data(validate, "SCI2")
plot_data(validate, "MPP")

# + {"deletable": true, "editable": true}
# Create final complete plot.
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(10, 5))

gs1 = gridspec.GridSpec(2, 1, wspace=0, hspace=0.04, left=0, right=0.27, bottom=0.08, top=0.98)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[1])


gs2 = gridspec.GridSpec(3, 1, wspace=0, hspace=0, left=0.32, right=0.62, bottom=0.08, top=0.98)
ax3 = fig.add_subplot(gs2[0])
ax4 = fig.add_subplot(gs2[1])
ax5 = fig.add_subplot(gs2[2])

gs3 = gridspec.GridSpec(3, 1, wspace=0, hspace=0, left=0.64, right=0.94, bottom=0.08, top=0.98)
ax6 = fig.add_subplot(gs3[0])
ax7 = fig.add_subplot(gs3[1])
ax8 = fig.add_subplot(gs3[2])




station_lats = [sta.latitude for net in filtered_inv for sta in net]
station_lngs = [sta.longitude for net in filtered_inv for sta in net]

ev_lats = [event.origins[0].latitude for event in cat]
ev_lngs = [event.origins[0].longitude for event in cat]


m_ortho = Basemap(resolution='c', projection='ortho', lat_0=10.,
                  lon_0=-150., ax=ax1)
m_ortho.drawcoastlines(color="0.3", linewidth=1.0)
m_ortho.drawcountries(color="0.8")
m_ortho.fillcontinents(color="0.9")

_xs, _ys = m_ortho(station_lngs, station_lats)
_xe, _ye = m_ortho(ev_lngs, ev_lats)

for sta, ev in itertools.product(zip(station_lngs, station_lats),
                                 zip(ev_lngs, ev_lats)):
    m_ortho.drawgreatcircle(*sta, *ev, linewidth=1.2,
                            color="0.6", alpha=0.5)

m_ortho.scatter(_xs, _ys, marker="v", s=120, zorder=10,
                color="k", edgecolor="w")
m_ortho.scatter(_xe, _ye, marker="o", s=120, zorder=10,
                color="k", edgecolor="w")


deg2m = 2 * np.pi * 6371 * 1000 / 360
m_local = Basemap(projection='aea',
                  resolution="i",
                  area_thresh=1000.0, lat_0=34.9,
                  lon_0=-118,
                  width=deg2m * 8, height=deg2m * 8,
                  ax=ax2)
m_local.drawmapboundary()
m_local.drawrivers(color="0.7")
m_local.drawcoastlines(color="0.3", linewidth=1.4)
m_local.drawcountries(color="0.4", linewidth=1.1)
m_local.fillcontinents(color="0.9")
m_local.drawstates(color="0.5", linewidth=0.9)

_x, _y = m_local(station_lngs, station_lats)
m_local.scatter(_x, _y, marker="v", s=140, zorder=10,
                color="k", edgecolor="w")


# Waveforms.
plt.sca(ax3)
plot_data(validate, "TIN", show=False)
plt.yticks(np.linspace(-1.5, 1.5, 7))

plt.sca(ax4)
plot_data(validate, "SCI2", show=False)
plt.yticks(np.linspace(-2, 2, 9))


plt.sca(ax5)
plot_data(validate, "MPP", show=False, legend=True)
plt.yticks(np.linspace(-2, 2, 9))


plt.sca(ax6)
stations = list(reference_shifts.keys())
delays = [reference_shifts[_i] for _i in stations]
plt.bar(np.arange(len(delays)) - 0.4, delays, 0.8, color="0.2")
plt.xticks(range(len(delays)), [""] * len(delays))
ax6.yaxis.tick_right()
ax6.yaxis.set_label_position("right")
plt.xlim(-1, 23)
plt.ylim(-0.8, 0.7)
plt.yticks([-0.5, 0, 0.5])
plt.ylabel("Average time shift [s]")
plt.text(0.97, 0.90, "Reference Time Shifts", transform=plt.gca().transAxes, va="top", ha="right")



plt.sca(ax7)
check_event(validate, ax=ax7)
ax7.yaxis.tick_right()
ax7.yaxis.set_label_position("right")
plt.yticks(np.linspace(-1.0, 1.0, 5))
plt.ylabel("Time shift deviation [s]")
plt.text(0.97, 0.90, "Validation Event", transform=plt.gca().transAxes, va="top", ha="right")




plt.sca(ax8)
check_event(application, ax=ax8, ticks=True)
ax8.yaxis.tick_right()
ax8.yaxis.set_label_position("right")
plt.yticks(np.linspace(-1.0, 1.0, 5))
plt.ylabel("Time shift deviation [s]")
plt.text(0.97, 0.90, "Test Event", transform=plt.gca().transAxes, va="top", ha="right")


plt.savefig("data_quality.pdf")
plt.show()
