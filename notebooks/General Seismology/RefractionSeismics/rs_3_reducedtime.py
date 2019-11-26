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

# <div style='padding: 0px ; background-size: cover ; border-radius: 5px ; height: 250px'>
#     <div style="float: right ; margin: 50px ; padding: 20px ; background: rgba(255 , 255 , 255 , 0.7) ; width: 50% ; height: 150px">
#         <div style="position: relative ; top: 50% ; transform: translatey(-50%)">
#             <div style="font-size: xx-large ; font-weight: 900 ; color: rgba(0 , 0 , 0 , 0.8) ; line-height: 100%">General Seismology</div>
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)"> Refraction Seismics: Reduced time </div>
#         </div>
#     </div>
# </div>

# Seismo-Live: http://seismo-live.org
#
#
# ##### Authors:
# * Heiner Igel ([@heinerigel](https://github.com/heinerigel))
#
# * Anokhi Shah
#
#
# ---

#
# In refraction seismology as well as in global seismology we often find travel-time diagrams where *reduced time* is used. In principle this means that the refraction arrival of interest is approximately horizontal in the travel-time diagram. This can be achieved by doing the following transformation
#
#
# $$t_{red} = t-\frac{\Delta}{v_{red}} \tag{1} $$ 
#
#
# where $v_{red}$ is the reduction velocity. To determine the real velocity from the travel-time diagrams in
# reduced form we choose a distance ${\Delta}_0$ and read the reduced travel time $tr_0$ from the diagram for the desired arrival. Then the velocity is calculated using
#
# $$v = \frac{\Delta_0}{{t_{r_0}} + \frac{\Delta_0}{v_{red}} - t_i}\tag{2}$$
#
# where $t_i$ is the intercept time and does not change when using reduced time. 
#
#
#

# + {"code_folding": [0], "hide_input": true}
# Import Libraries (PLEASE RUN THIS CODE FIRST!)
import matplotlib.pyplot as plt
import numpy as np
# %pylab inline
pylab.rcParams['figure.figsize'] = (10, 7)

# + {"code_folding": [0]}
# Initialization parameters - layered velocity model
v1= 3.5                       #km/s   #Can be changed
v2= 5                       #km/s   #Can be changed
v3= 8                       #km/s   #Can be changed
h1= 10                       #km     #Can be changed
h2= 25                       #km     #Can be changed

D=np.linspace(0,500,1000)   # Variable distance/Delta over x axis

i12= arcsin(v1/v2)
i13= arcsin(v1/v3)
i23= arcsin(v2/v3)

tD = D/v1                                                                  #Direct Wave
trefr = ((2*h1*cos(i12)/v1)+ D/v2)                                         #Refracted Form
tRF = ((((2*h1*cos(i13))/v1) + (2*h2*cos(i23)/v2) + (D/v3))- (D/v3))       #Reduced form

# + {"code_folding": [0]}
# Initialization plotting

plt.plot(D, tD, color='green',label='Direct Wave from Layer 1 => 1/v1' )
plt.plot(D, trefr, color='Blue', label='Refracted Wave from layer 2 => 1/v2')
#plt.plot(D, ((2*h1*cos(i13))/v1) + (2*h2*cos(i23)/v2) + (D/v3), color ='Red',label='Refracted Wave from layer 3 => 1/v3' )
plt.plot(D, tRF, color ='orange',label='Reduced Form')
                           #Naming the axis
plt.xlabel('Distance in KM ')
plt.ylabel('Time in sec')
plt.title('Travel Time diagram for the 3 layer case in reduced form')
                          #setting limits for x and y axis
plt.ylim((0,90))
plt.xlim((0, 300))
plt.legend()
plt.show()
