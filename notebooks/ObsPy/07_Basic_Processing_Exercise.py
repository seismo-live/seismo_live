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
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)">Downloading/Processing Exercise</div>
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

# For the this exercise we will download some data from the Tohoku-Oki earthquake, cut out a certain time window around the first arrival and remove the instrument response from the data.

# %matplotlib inline
from __future__ import print_function
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 12, 8

# The first step is to download all the necessary information using the ObsPy FDSN client. **Learn to read the documentation!**
#
# We need the following things:
#
# 1. Event information about the Tohoku-Oki earthquake. Use the `get_events()` method of the client. A good provider of event data is the USGS.
# 2. Waveform information for a certain station. Choose your favorite one! If you have no preference, use `II.BFO` which is available for example from IRIS. Use the `get_waveforms()` method.
# 3. Download the associated station/instrument information with the `get_stations()` method.



# Have a look at the just downloaded data.



# ## Exercise
#
# The goal of this exercise is to cut the data from 1 minute before the first arrival to 5 minutes after it, and then remove the instrument response.
#
#
# #### Step 1: Determine Coordinates of Station



# #### Step 2: Determine Coordinates of Event



# #### Step 3: Calculate distance of event and station.
#
# Use `obspy.geodetics.locations2degree`.



# #### Step 4: Calculate Theoretical Arrivals
#
# ```python
# from obspy.taup import TauPyModel
# m = TauPyModel(model="ak135")
# arrivals = m.get_ray_paths(...)
# arrivals.plot()
# ```



# #### Step 5: Calculate absolute time of the first arrivals at the station



# #### Step 6:  Cut to 1 minute before and 5 minutes after the first arrival



# #### Step 7: Remove the instrument response
#
# ```python
# st.remove_response(inventory=inv, pre_filt=...)
# ```
#
# ![taper](images/cos_taper.png)



# ## Bonus: Interactive IPython widgets

# +
from IPython.html.widgets import interact
from obspy.taup import TauPyModel

m = TauPyModel("ak135")

def plot_raypaths(distance, depth, wavetype):
    try:
        plt.close()
    except:
        pass
    if wavetype == "ttall":
        phases = ["ttall"]
    elif wavetype == "diff":
        phases = ["Pdiff", "pPdiff"]
    m.get_ray_paths(distance_in_degree=distance,
                    source_depth_in_km=depth,
                    phase_list=phases).plot();
    
interact(plot_raypaths, distance=[0, 180],
         depth=[0, 700], wavetype=["ttall", "diff"]);
