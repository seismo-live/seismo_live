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
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)"> Refraction Seismics: Three layer case</div>
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

# This notebook presents the refraction of a seismic wave in a tree layer medium. 
#
# The three layer case is important for many realistic problems, particularly for near surface seismics, where often a low velocity weathering layer is on top of the bedrock. In principle we follow the same reasoning as a two-layer case, but through the additional layer there is more algebra involved. We have to introduce a slightly different nomenclature to take into account the different layers. The incidence angles will have two indices, the first index stands for the layer in which the angle is defined and the last index corresponds to the layer in which the ray is refracted, as shown in the Figure. The equation for the direct waves is of course the same as in the two-layer case. The same is true for the refraction from layer 2 but we show it to demonstrate the nomenclature.

# ![theelayercase.png](attachment:theelayercase.png)

# #### The refraction from layer 2
#
# The arrival time $t_2$ of the refraction from layer $2$ is given by
#
# $$4t_2 = \frac{2h_1\cos{i_{12}}}{v_1} + \frac{\Delta}{v_2} = t^{12} + \frac{\Delta}{v_2} \tag{1}$$
#
# and, using the intercept time from the diagram from the travel-time for the two-layer case, will allow us to determine the depth $h_1$ of the topmost layer.
#
# #### The refraction from layer 3 
#
# Due to Snell's law we have
#
# $$\frac{sin{i_{13}}}{v_1} = \frac{sin{i_{23}}}{v_2} = \frac{sin{i_{33}}}{v_3} \tag{2}$$
#
# we use this relation and basic trigonometry to derive the arrival time $t_3$ of the refracted wave in layer 3
#
# $$t_3 = \underbrace{\frac{2h_1\cos{i_{13}}}{v_1} + \frac{2h_2\cos{i_{23}}}{v_2}}_{t^{i3}} + \frac{\Delta}{v_3} = t^{i3} + \frac{\Delta}{v_3} \tag{3}$$
#
# and again this is a straight line with the intercept time $t^{i3}$ which can be read from the travel time diagram.

# ### Determining the velocity depth model for the 3-layer case
#
# As in the two-layer case our data is a diagram with the travel-times of the direct wave, the refraction from layer $2$ and the refraction from layer $3$, where we were able to read the arrivals in the seismograms. To determine the velocities and the thicknesses of layers $1$ and $2$ first the velocities $v_{1-3}$ from the slopes $\frac{1}{v_{1-3}}$ in the travel-time diagram are set. Then the intercept time $t^{i2}$ for the refraction from layer $2$ are red. To determine thickness $h_1$, using equation the Snell's law (Equation (1) in [Two-layer case notebook](rs_twolayercase.ipynb)) we obtain
#
# $$h_1 = \frac{v_1t^{i2}}{2\cos{i_{12}}} \quad   \text{where} \quad  i_{12}\ = \arcsin{\frac{v_1}{v_2}} \tag{4}$$
#
#
# Then reading the intercept time $t^{i3}$ for the refraction from layer $3$ and calculating with the already determined values $h_1$ an intermediate intercept time $t^*$
#
# $$t^* = t^{i3}-\frac{2h_1\cos{i_{13}}}{v_1}\quad   \text{where} \quad  i_{13} = \arcsin\frac{v_1}{v_2} \tag{5}$$
#
# And finally using $t^*$ calculate the thickness $h_2$ of layer $2$
#
# $$h_2 = \frac{v_2t^*}{2cos{i_{23}}} \quad   \text{where} \quad  i_{23} = \arcsin\frac{v_2}{v_3} \tag{6}$$

# + {"code_folding": [0]}
# Import Libraries (PLEASE RUN THIS CODE FIRST!)
import matplotlib.pyplot as plt
import numpy as np
# %pylab inline
pylab.rcParams['figure.figsize'] = (10, 7)

# + {"code_folding": [0]}
#Parameters initialization
v1= 3.5                       #km/s   #Can be changed
v2= 5                       #km/s   #Can be changed
v3= 8                       #km/s   #Can be changed
h1= 10                       #km     #Can be changed
h2= 25                       #km     #Can be changed

D=np.linspace(0,500,1000)   # Variable distance/Delta over x axis

i12= arcsin(v1/v2)
i13= arcsin(v1/v3)
i23= arcsin(v2/v3)
# -

# The predetermined model velocities are
# * $v_1$ =3.5km/s 
# * $v_2$ =5km/s 
# * $v_3$ =8km/s
#
# And the layer thicknesses are
# * $h_1$ =10km 
# * $h_2$ =25km

# + {"code_folding": [0]}
#Plotting-Do not change

plt.plot(D, D/v1, color='green',label='Direct Wave from Layer 1 => 1/v1' )
plt.plot(D, ((2*h1*cos(i12)/v1)+ D/v2), color='Blue', label='Refracted Wave from layer 2 => 1/v2')
plt.plot(D, ((2*h1*cos(i13))/v1) + (2*h2*cos(i23)/v2) + (D/v3), color ='Red',label='Refracted Wave from layer 3 => 1/v3' )

annotation_string = r"$t^{i3}$" % (D)
annotation_string += "\n"
annotation_string += "\n"
annotation_string += r"$t^{i2}$" % (D)
plt.annotate(annotation_string, xy=(1, 5))

#Naming the axis
plt.xlabel('Distance in KM ')
plt.ylabel('Time in sec')

#setting limits for x and y axis
plt.ylim((0,90))
plt.xlim((0, 300))
plt.legend()
plt.show()
# -


