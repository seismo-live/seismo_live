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
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)"> Refraction Seismicis: Two layer case</div>
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
# ---

# This notebook presents how a seismic wave is refracted in a two layer medium. 
#
# We consider the case where a layer with thickness $h$ and velocity $v_1$ is situated over a halfspace with velocity $v_2$. A receiver is located at a distance $\Delta$ from the source, which itself is located at the surface. From the whole range of seismic signals we
# measure, if a seismic source is generating energy (e.g. an explosion), here we will only consider the direct waves, reflections and refractions but not take into account multiple reverberations which would be recorded in nature (but often neglected in the processing). 
# The geometry of the problem looks like this:

# ![Untitled.png](attachment:Untitled.png)

# Before we try to determine the structure from observed travel times we have to understand the forward problem:
# how can we determine the travel time of the three basic rays as a function of the velocity structure and the distance from the source. The most important ingredient we need is Snell's law relating the incidence angle $i$ in layer 1 with velocity $v_1$ to the transmission angle i in layer 2 with velocity $v_2$, both
# angles are measured with respect to the vertical. 
#
# $$\frac{\sin{i_1}}{v_1} = \frac{\sin{i_2}}{v_2} \tag{1}$$
#
# Then, we can derive the arrival times for the three types separately:

# #### The direct wave
#
# In a layered medium the direct wave travels straight along the surface with velocity $v_1$. At
# distance $\Delta$ the travel time is
# $$t_{dir} = \frac{\Delta}{v_1} \tag{2}$$
#
# #### The reflected wave
#
# To calculate the reflected wave we need to do a little geometry. The length of the path the ray travels in layer 1 is obviously related to the distance in a non-linear way. The travel time for the reflection is given by
#
# $$t_{refl} = \frac{2}{v_1}\sqrt{\bigg({{\frac{\Delta}{2}\bigg)^2}+ {h^2}}} \tag{3}$$
#
# In refraction seismology this arrival is often of minor interest, as the distances are so large that the reflected wave has merged with the direct wave, which has the form of a hyperbola.

# #### The refracted wave
# As we can easily see from the curve above the refracted wave needs a more involved treatment. Refracted waves
# correspond to energy which propagates horizontally in medium $2$ with the velocity $v_2$. This can only happen if the emergence angle $i_2$ is $90^\circ$, i.e. the critical angle ${i_c}$.
#
# $$\frac{sin({i_c})}{v_1} = \frac{sin{90^\circ}}{v_2} = \frac{1}{v_2}\implies \sin{i_c} = \frac{v_1}{v_2} \tag{4}$$
#
# In order to calculate the travel time we need to consider rays which impinge on the discontinuity with angle ${i_c}$. From elementary geometry follows that the arrival time $t_{refr}$ of the refracted wave as a function of distance $\Delta$ is given by
#
# $$t_{refr} = \frac{2h\cos{i_c}}{v_1} + \frac{\Delta}{v_2} = {t^i}_{refr} + \frac{\Delta}{v_2} \tag{5}$$
#
# which is a straight line which crosses the time axis $\Delta=0$ at the intercept time ${t^i}_{refr}$ and has a slope $\frac{1}{v_2}$.
#
# ### Travel time curves (the forward problem)
# Now we can put everything together and calculate, for a given velocity model, the arrival times and plot them in a
# travel-time diagram. 

# + {"code_folding": [0]}
# Import Libraries (PLEASE RUN THIS CODE FIRST!)
import matplotlib.pyplot as plt
import numpy as np
# %pylab inline
pylab.rcParams['figure.figsize'] = (10, 7)

# + {"code_folding": []}
# Initialization parameters
v1= 5                       #km/s   #Can be changed
v2= 8                       #km/s   #Can be changed
h= 30                       #km     #Can be changed
ic= np.arcsin(v1/v2)              # critical angle
D=np.linspace(0,500,1000)   # Variable distance/Delta over x axis

tRl = 2/v1*sqrt((D/2)**2 + h**2,)
tD = D/v1
tRr = ((((2*h*cos(ic)))/v1) + (D/v2))
# -

# The predetermined model parameters are chosen to correspond to a very simple model of crust and upper mantle and the discontinuity would be the Moho.
# The distance at which the refracted arrival overtakes the direct arrival can be used to determine the layer depth.
# According to ray theory there is a minimal distance at which the refracted wave can be observed, this is called the critical distance. 
#
# * Thickness of the layer: $ \Delta = 30$ km
# * Velocity of the medium $1$: $v_1 = 5$ km/s
# * Velocity of the medium $2$: $v_2 = 8$ km/s
#
#

# + {"code_folding": [0]}
# Plotting

plt.plot(D, tRl, color='green',label='Reflected Wave') 

plt.plot(D, tD, color='blue', label='Direct Wave' )

plt.plot(D, tRr, color='red',  label='Refracted Wave')        

#Naming the axis
plt.xlabel('Distance (km) ')
plt.ylabel('Time (s)')
plt.title('Figure 2: Travel-time diagram for two-layer case')


#setting limits for x and y axis
plt.ylim((0,120))
plt.xlim((0,500))

plt.legend()
plt.show()
# -

# ### Critical distance and overtaking distance
#
# Two concepts are useful when determining the depth of the top layer. The critical distance is the distance at which the refracted wave is first observed according to ray theory, in real life it is observed already at smaller distances, this is due to Innite-frequency effects which are not taken into account by standard ray theory. This critical distance $\Delta_c$ is
#
# $$\Delta_c = 2h\tan{i_c} \tag{6}$$
#
# where the critical angle $i_c$ is given by equation (4). If we equate the arrival time of the direct wave and the refracted wave and solve for the distance we obtain the overtaking distance. It is given by
# $$\Delta_u = 2h\sqrt{\frac{v_2 + v_1}{v_2 - v_1}} \tag{7}$$

# ### Exercise
# #### Determining the structure from travel-time diagrams: the inverse problem.
#
# Determine the velocity depth model from the observed travel times (Figure 2). 
# * Determine $v_1$ of the direct wave.
# * Determine $v_2$ of the refracted wave.
# * Calculate the critical angle from $v_1$ and $v_2$.
# * Read the intercept time $t_i$ from the travel-time diagram.
# * Determine the depth $h$ using equation (5) or read the overtaking distance from the travel-time diagram, and calculate $h$ using equation (6)




