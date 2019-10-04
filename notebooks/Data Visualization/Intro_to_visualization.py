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

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# <div style='background-image: url("../share/images/header.svg") ; padding: 0px ; background-size: cover ; border-radius: 5px ; height: 250px'>
#     <div style="float: right ; margin: 50px ; padding: 20px ; background: rgba(255 , 255 , 255 , 0.7) ; width: 50% ; height: 150px">
#         <div style="position: relative ; top: 50% ; transform: translatey(-50%)">
#             <div style="font-size: xx-large ; font-weight: 900 ; color: rgba(0 , 0 , 0 , 0.8) ; line-height: 100%">Scientific Visualization with Python</div>
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)">A super quick crash course</div>
#         </div>
#     </div>
# </div>

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# Seismo-Live: http://seismo-live.org
#
# ##### Authors:
# * Stephanie Wollherr ([@swollherr](https://github.com/swollherr))
#
# ---

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# This notebook introduces some basic plotting examples using matplotlib.

# + {"deletable": true, "editable": true}
#we need the following packages, always execute this cell at the beginning

#plots inside the notebook
# %matplotlib inline

#you can use these libraries by refering to their appreviation plt., np. or pd.
#basic plotting library
import matplotlib.pyplot as plt

#scientifc computing library
import numpy as np

#data analysis tool
import pandas as pd

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# # Reading in your data
#
# To handle data in python we first have to read it in.
# ### Simple ascii/txt/csv files.
#
# #### loadtxt by numpy

# + {"deletable": true, "editable": true}
#read in a seismogram
#4 columns with 
#time North-South East-West Up-Down
time, ns, ew, ud = np.loadtxt('data/station_1.dat').T
print(time)
print(ns)

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# usefull parameters for loadtxt:
# - comments : The characters or list of characters used to indicate the start of a comment; default: ‘#’.
# - skiprows : Skip the first skiprows lines; default: 0.
# - usecols : Which columns to read, with 0 being the first. For example, usecols = (1,4,5) will extract the 2nd, 5th and 6th columns. The default, None, results in all columns being read.

# + {"deletable": true, "editable": true}
#in action: we only need the time series and the North-South component
time, ns = np.loadtxt('data/station_1.dat', usecols=(0,1)).T
print(time)
print(ns)

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# #### read_csv by pandas
# much faster than loadtxt, in particular for large files

# + {"deletable": true, "editable": true}
#read it in without any specifications
data_pd = pd.read_csv('data/station_1.dat')
print (data_pd)

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# **It reads in everything!**
#
# So we have to set a couple of parameters. Possible parameters that we can set:
# - sep : string; delimiter to use; default ‘,’ 
# - header : integer or list of integers; row number(s) to use as the column names and the start of the data
# - names :  List of column names to use. If file contains no header row, then you should explicitly pass header=None; array-like; default None.
# - usecols : Return a subset of the columns; array-like or callable; default None. 
# - skiprows : Line numbers to skip (0-indexed) or number of lines to skip (int) at the start of the file; list-like or integer or callable; default None. 
#
# ... and many more: full list available at https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html
#
# Let's try this again:

# + {"deletable": true, "editable": true}
#files uses tab-seperation, so we'll let read_csv know
#names can be used for header names, header can also be read in but now we're skipping them with comment='#'
data_pd = pd.read_csv('data/station_1.dat', sep = '\t', comment='#', names=['time','NS','EW', 'UD'])
print (data_pd)

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# Now we can nicely refer to our data with:

# + {"deletable": true, "editable": true}
print('time')
print (data_pd['time'])

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# ### Other data formats
#
# There are other possibilties to read in your data when it's not in ascii format.
#
# For example, we will later use netcdf-based data formats in the second tutorial.
#
# More packages for handling data input:
# - netcdf4 package (Network Common Data Form)
# - read in hdf with panda (Hierarchical Data Format)
#
# **What kind of data format are you using?**
#
#
#

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# ## Plotting your data - Simple Plots
#
# Matplotlib is a widely used plotting library that we will use here for our first simple plots.
#
# 1. Single Figures
# 2. Axis labels, titles
# 3. Subplots
# 4. Styles
# 5. Scatter plots
# 6. Other types of plots
# 7. How to save your plots
#

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# ## 1. Single figures
# Call the plotting function together with its package that we called 'plt'.
#
# The plot gets visible by inserting plt.show() at the end.

# + {"deletable": true, "editable": true}
plt.plot(data_pd['time'], data_pd['NS'])
plt.show()

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# ## 2. Labels and titles
# Let's insert more information!
# - label the axis
# - insert a legend
# - give a title

# + {"deletable": true, "editable": true}
#default label is the name of the array, but you can also label it with a own name

plt.plot(data_pd['time'], data_pd['NS'])
#plt.plot(data_pd['time'],data_pd['NS'], label='North-South')

plt.legend()

#axis labels
plt.xlabel('time [s]')
plt.ylabel('acceleration cm/s²')

#title
plt.title('Station LUC')

plt.show()

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# ## 3. Subplots
#
# We now want to plot all three components of the seismogram in one single plot.
#
# Subplots are structured as follows:
# <img src="images/subplots.png" style="width:70%"></img>
#
# We will now introduce three axis in one figure.

# + {"deletable": true, "editable": true}
#create a figure f and three subplots
f, (ax1, ax2, ax3) = plt.subplots(3, 1)

ax1.plot(data_pd['time'], data_pd['NS'])
#for axis properties we use set_*
ax1.set_xlabel('time [s]')
ax1.set_ylabel('acceleration cm/s²')
ax1.legend()

ax2.plot(data_pd['time'], data_pd['EW'])
ax2.set_xlabel('time [s]')
ax2.set_ylabel('acceleration cm/s²')
ax2.legend()

ax3.plot(data_pd['time'], data_pd['UD'])
ax3.set_xlabel('time [s]')
ax3.set_ylabel('acceleration cm/s²')
ax3.legend()


#plot title over all subplots
f.suptitle('station LUC', size=20)

#needs to be shifted when tight_layout is used
f.subplots_adjust(top=0.5)
#makes all axis labels visible
f.tight_layout()

f.show()

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# Beautify the plot!
# - plots can share axis, plots can even share boxes
# - focus on the area of interest

# + {"deletable": true, "editable": true}
#share x axis
f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)

ax1.plot(data_pd['time'], data_pd['NS'])
ax1.set_ylabel('acc. cm/s²')
ax1.legend()

ax2.plot(data_pd['time'], data_pd['EW'])
ax2.set_ylabel('acc. cm/s²')
ax2.legend()

ax3.plot(data_pd['time'], data_pd['UD'])
ax3.set_ylabel('acc. cm/s²')
ax3.set_xlabel('time')
ax3.legend()

#share the box
plt.subplots_adjust(hspace=0)

#reduce number of ticks
plt.locator_params(axis='both', numtick=8)

#focus on the time where the signal is
plt.xlim((10.0,40.0))

#plot title over all subplots
f.suptitle('station LUC', size=20)


plt.show()

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# ## 4. Styles
#
# You can use different style option to render your plot.
#
# Full documentation available here: https://matplotlib.org/devdocs/gallery/style_sheets/style_sheets_reference.html

# + {"deletable": true, "editable": true}
print(plt.style.available)

# + {"deletable": true, "editable": true}
# choose a style from above
plt.style.use('default')

#you can also combine different styles
#plt.style.use(('fivethirtyeight', 'seaborn-white', 'seaborn-pastel'))

f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, sharey=True)

ax1.plot(data_pd['time'],data_pd['NS'])
ax1.set_ylabel('acc. cm/s²')

ax2.plot(data_pd['time'],data_pd['EW'])
ax2.set_ylabel('acc. cm/s²')

ax3.plot(data_pd['time'],data_pd['UD'], label='data1')
ax3.set_ylabel('acc. cm/s²')
ax3.set_xlabel('time')
ax3.legend()


#reduce number of ticks
plt.locator_params(axis='both',numtick=8)

#focus on the time where the signal is
plt.xlim((10.0,40.0))

#plot title over all subplots
f.suptitle('station LUC', size=20)


plt.show()

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# or you can change the **line style and colors**.

# + {"deletable": true, "editable": true}
#read in another dataset
data2_pd = pd.read_csv('data/station_2.dat', sep = '\t', comment='#', names=['time','NS','EW', 'UD'])

# + {"deletable": true, "editable": true}
f, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True, sharey=True)

ax1.plot(data_pd['time'], data_pd['NS'])
ax1.plot(data2_pd['time'], data2_pd['NS'], ls='-.')
ax1.set_ylabel('acc. cm/s²')

ax2.plot(data_pd['time'], data_pd['EW'])
ax2.plot(data2_pd['time'], data2_pd['EW'], ls='--')
ax2.set_ylabel('acc. cm/s²')

ax3.plot(data_pd['time'], data_pd['UD'], label='station 1')
ax3.plot(data2_pd['time'], data2_pd['UD'], ls=':', label='station 2')
ax3.set_ylabel('acc. cm/s²')
ax3.set_xlabel('time')
ax3.legend()


#reduce number of ticks
plt.locator_params(axis='both',numtick=8)

#focus on the time where the signal is
plt.xlim((10.0,40.0))

#plot title over all subplots
f.suptitle('Landers earthquake', size=20)


plt.show()

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# ## 5. Scatter Plots

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# Using scatter plots we can show discrete data, for example measurements in dependence of two (or even three) dimensions.
#
# Full documentation: https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.scatter.html
#

# + {"deletable": true, "editable": true}
# For this example we first create some random input.
# You can use your own dataset by reading in your data first
x = np.random.randn(100)
y = x + np.random.randn(100) + 10

#-------
# Insert linear fit

#from scipy import stats
#slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
#line = slope*x+intercept
#we can add some oppacity with alpha (0 to 1)
#plt.plot(x, line, alpha=0.5)

#-------
plt.xlabel('x')
plt.ylabel('y')
plt.title('Example Scatter Plot')

plt.scatter(x,y)


plt.show()

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# Additionally, we can use color and size as a third dimension of information in our scatter plot.
#
# Available colorbars: https://matplotlib.org/users/colormaps.html

# + {"deletable": true, "editable": true}
#create a third dimension, for example temperature
z = np.random.randn(100)*70

#plt.scatter(x,y, cmap='viridis', c=z)

#even better to spot the difference: 
#change the size according to the third data dimension
#colorbar can be restricted by vmin and vmax
plt.scatter(x, y, cmap='viridis', c=z, s=z, vmin=-40, vmax=140)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Example Scatter Plot')

plt.colorbar()

#set colorbar axis to customized range
#plt.set_cmap([0,150])

plt.show()

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# ### Additional options on markers and labels
#
# You can customize your scatter plot by adding different markers for different datasets and labels.
#
# Markerstyles can be found here: https://matplotlib.org/api/markers_api.html

# + {"deletable": true, "editable": true}
#create a second data set
x_2 = np.random.randn(100)
y_2 = x_2 + np.random.randn(100) + 10
z_2 = np.random.randn(100)*70

#shades of red with z=color itensity
plt.scatter(x, y, cmap='Reds', c=z, label='dataset 1')

#shades of blue with z=color intensity
plt.scatter(x_2, y_2, cmap='Blues', c=z_2, marker ='v', label='dataset 2')


plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.title('Difference day and night')
plt.show()

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# ### 3D scatter plots
#
# You can also use a  3D figure to plot your three dimensional data set.

# + {"deletable": true, "editable": true}
# we need the following package
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#colored by the z values
ax.scatter(x, y, z, c=z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.title('3D scatter plot')
plt.show()

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# ## 6. Other types of plots
#
# Some additional graphics and inspiration.
#
# ### Bar plots
#
#

# + {"deletable": true, "editable": true}
#Generates
n = 12
#creates a array ranging from 0 to 11
X = np.arange(n)

#n random number, uniform distribution
Y1 = np.random.uniform(0.5, 1.0, n)
Y2 = np.random.uniform(0.5, 1.0, n)

#set a face and edge color
plt.bar(X, +Y1, facecolor='#9999ff')
plt.bar(X, -Y2, facecolor='#ff9999')

for x,y in zip(X,Y1):
    #annotations
    #shift the values slightly above the bars
    plt.text(x+0.1, y+0.05, '%.2f' % y, ha='center', va= 'bottom')

for x,y in zip(X,Y2):
    plt.text(x+0.1, -y-0.05, '%.2f' % y, ha='center', va= 'top')

plt.xlim(-.5, n)
#remove ticks on x axis
plt.xticks([])

plt.ylim(-1.25, +1.25)
#or fully remove the ticks/labels
plt.yticks([])

plt.title('Histogram example')

plt.show()


# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# ### Seaborn
#
# Seaborn is a package dedicated to statistical graphics.
#
# More general information: http://seaborn.pydata.org/examples/
#
# Jointplots combine scatter plots with distribution plots along the two axis.
#
# More information about jointplots: https://seaborn.pydata.org/generated/seaborn.jointplot.html

# + {"deletable": true, "editable": true}
#same random data set as before for the scatter plot
#this time with seaborn
import seaborn as sns

x_2 = np.random.randn(100)
y_2 = x_2 + np.random.randn(100) + 10

#data needs to put into this format
data = pd.DataFrame({"x": x_2, "y": y_2})

#scatter plot with distribution along x and y coordinate
#first plot: default
sns.jointplot(x=x_2, y=y_2, data=data)

#but you can also play around with the "kind" option
#second plot: hex + changed color
sns.jointplot(x=x_2, y=y_2, data=data, kind="reg", space=0, color="r")

#third plot: kde + changed color
sns.jointplot(x=x_2, y=y_2, data=data, kind="kde", space=0, color="g")

plt.show()

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# Some more inspiration

# + {"deletable": true, "editable": true}
#Error bar plots

sns.set(style="ticks")

#Load the example tips dataset
tips = sns.load_dataset("tips")

#Draw a nested boxplot to show bills by day and sex
sns.boxplot(x="day", y="total_bill", hue="sex", data=tips, palette="PRGn")

#This setting removes the borders to minimalize the figure
sns.despine(offset=10, trim=True)

plt.show()

# + {"deletable": true, "editable": true}
# Heatmaps
sns.set()

# Load the example flights dataset and conver to long-form
flights_long = sns.load_dataset("flights")
flights = flights_long.pivot("month", "year", "passengers")

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(9, 6))

#main command
sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)
plt.title('Passengers')

plt.show()

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# ## 7. How to save your plots

# + {"deletable": true, "editable": true, "cell_type": "markdown"}
# You can easily save your nice figures by using
#
# ```python
# plt.savefig('my_figure.png')
# ```
# before calling 
#
# ```python
# plt.show()
# ```
#
# The resolution can be increase by increasing dpi values or saving in pdf or svg format.
#
# ```python
# plt.savefig('my_figure.png', dpi=800)
# plt.savefig('my_figure.svg')
# ```
