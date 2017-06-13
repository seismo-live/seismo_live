# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:20:16 2016

@author: tbartholomaus

A few custom functions that make it easier to work with obspy UTCDateTimes
"""

from obspy import UTCDateTime
import numpy as np
#import datetime as dt
from matplotlib.dates import date2num

#print('Hello')

# Round down a given UTCDatetime to the beginning of the day (midnight) of the given day.
def UTCfloor(t):
    t_floor = UTCDateTime(t.year, t.month, t.day)
    
    return t_floor

   
# Round up a given UTCDatetime to the beginning of the next day (midnight) from the given day.
def UTCceil(t):
    t_ceil = UTCDateTime(t.year, t.month, t.day + 1)
    
    return t_ceil

# Transform an np.array of UTCDateTimes into datenums, for the purpose of plotting
def UTC2dn(t):
    t_datenum = np.empty(len(t))
    for i in range(len(t)):
        t_datenum[i] = date2num(t[i].datetime)

    return t_datenum
