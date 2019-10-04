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
#             <div style="font-size: xx-large ; font-weight: 900 ; color: rgba(0 , 0 , 0 , 0.8) ; line-height: 100%">Instaseis Tutorial</div>
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)">Part 3: Plot a Record Section</div>
#         </div>
#     </div>
# </div>

# Seismo-Live: http://seismo-live.org
#
# ##### Authors:
# * Martin van Driel ([@martinvandriel](https://github.com/martinvandriel))
# * Lion Krischer ([@krischer](https://github.com/krischer))
# ---

# <img style="width:50%" src="images/logo.png">

# ## Advanced Exercise 1: Plot Record Section
#
# Use `Instaseis` to calculate a record section of your choice.
#
# ![image](./images/record_section.png)
#
# #### Notes
#
# 1. ObsPy Trace objects provide a function `times()` to conveniently get the time axis for plotting and the `normalize()` function to normalize seismograms to a common amplitude.
# 2. Use high pass filtering and deep sources if you want to enhance body waves
# 3. `np.linspace()` can help to generate equidistant stations
# 4. `obspy.taup.TauPyModel` can be used to compare with ray theoretical arrivals

# -----
#
# Basic lines to set up the notebook and some paths.

# %matplotlib inline
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import obspy
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 8)

# Import Instaseis and open the database:

import instaseis
db = instaseis.open_db("data/database")

# -----

# ## Solution:


