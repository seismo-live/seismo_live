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

# <div style='padding: 0px ; background-size: cover ; border-radius: 5px ; height: 250px'>
#     <div style="float: right ; margin: 50px ; padding: 20px ; background: rgba(255 , 255 , 255 , 0.7) ; width: 50% ; height: 150px">
#         <div style="position: relative ; top: 50% ; transform: translatey(-50%)">
#             <div style="font-size: xx-large ; font-weight: 900 ; color: rgba(0 , 0 , 0 , 0.8) ; line-height: 100%">General Seismology</div>
#             <div style="font-size: large ; padding-top: 20px ; color: rgba(0 , 0 , 0 , 0.5)"> Refraction Seismics: Inclined Two-layer case </div>
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

# The other notebooks presented in this section only considered plane layers with no structural variation along the profile. In this one, we consider the case where a high-velocity layer is inclined with an inclination angle α (see Figure). 
#
# The most important difference to the 2-layer and 3-layer cases is that we now perform two experiments, one shooting at the near end and one shooting at the far end of the region of interest. For the previous examples, due to symmetry, we would have observed the same travel time curves, otherwise for the this notebook of an inclined layer this is no longer the case. 

# ![Reduce%20Time%20image.png](attachment:Reduce%20Time%20image.png)

# Let us develop the forward problem, i.e. calculating the travel times of the direct and refracted waves for a given
# model. With seismic velocities $v_1$ and $v_2$ and inclination angle ${\alpha}$ the travel time of the refracted waves are
#
# $${t^-}_{refr} = \frac{2h^-\cos{i_c}}{v_1} + \frac{\sin{(i_c+\alpha)}}{v_1}\Delta = {t^-}_i + \frac{1}{{v^-}_2}\Delta\tag{1}$$
#
# $${t^+}_{refr} = \frac{2h^+\cos{i_c}}{v_1} + \frac{\sin{(i_c-\alpha)}}{v_1}\Delta = {t^+}_i + \frac{1}{{v^+}_2}\Delta\tag{2}$$
#
#
# where the (-) sign stands for the refracted arrival with smaller intercept time and the (+) sign for the refraction with larger intercept time, $i_c$ is the critical angle at the interface and $h^+$ and $h^-$ are defined according to the Figure above.
#
# The arrival of the direct wave, as in the other cases, is at time $t_{dir}={\Delta}/v_1$. To determine the model properties from the observed arrival times, the inverse problem, first the velocities $v_1$ and $v_2^{+/-}$ are determined from the slopes in the travel-time diagram, using the following equations.
#
# $$\sin({i_c + \alpha}) = \frac{v_1}{{v_2^-}} \implies i_c + \alpha = \arcsin\frac{v_1}{{v_2^-}} \tag{3}$$
#
#
# $$\sin({i_c - \alpha}) = \frac{v_1}{{v_2^+}} \implies i_c - \alpha = \arcsin\frac{v_1}{{v_2^+}} \tag{4}$$
#
# Solving the equations (3) and (4) leads us
#
# $$\frac{({i + \alpha}) + (i-\alpha)}{2} = i
# \implies
# v_2 = \frac{v_1}{\sin{i}} \tag{5}$$
#
# The equation (5) reduces to,
#
# $$\frac{({i + \alpha}) - (i-\alpha)}{2} = \alpha \tag{6}$$
#
# Then, reading the intercept times $t_i^+$ and $t_i^-$ from the travel time diagram one can determine the distances from the layer interface as
#
# $$h^- = \frac{v_1{t^-}_i}{2\cos{i_c}} \tag{7}$$
#
# $$h^+ = \frac{v_1{t^+}_i}{2\cos{i_c}} \tag{8}$$
#
# We can plot graphically the layer interface by drawing circles around the profile ends with the corresponding heights $h^{+/-}$ and tangentially connecting the circles at depth.

# + {"code_folding": [0], "hide_input": true}
# Import Libraries (PLEASE RUN THIS CODE FIRST!)
import matplotlib.pyplot as plt
import numpy as np
# %pylab inline
pylab.rcParams['figure.figsize'] = (10, 7)

# + {"code_folding": [0]}
# Initialization Parameters
v1= 1.2                         #km/s   #Can be changed
v2= 4                           #km/s   #Can be changed
h1= 0.6                         #km     #Can be changed
h2= 0.5                           #km   #Can be changed
A = 8./180*pi                   #  degree #can be changed

D=np.linspace(0,1500,1000)   # Variable distance/Delta over x axis

ic= cos(v1/v2)              # critical angle

tD = D/v1                   #Direct Wave time
th1 = (2*h1*cos(ic))/v1 + (sin(ic+A)/v1)*D # Time Refracted wave at h1
th2 = (2*h2*cos(ic)/v1) + (sin(ic-A)/v1)*D # Time Refracted wave at h2
# -

# The travel time curves that will be observed for a model with ${\alpha} = 8{^\circ}$, $v_1 =1.2$ km/s and $v_2= 4$ km/s is plotted below. 

# + {"code_folding": [0]}
# Plotting

plt.plot(D, tD, color='green',label='Direct Wave from Layer 1 => 1/v1' )
plt.plot(D, th1, color = 'red')
plt.plot(np.max(D)-D, th1, color = 'red', label='Refracted wave at h1')
plt.plot(D, th2, color = 'blue', label = 'Refracted wave at h2')
plt.plot(np.max(D)-D, th2, color = 'blue')



plt.legend()
#plt.ylim((0,0.9))
plt.xlim((0, 1500))
xlabel('Distance(m)')
ylabel('Time(s)')
plt.show()
# -

# ### Exercise: The inverse problem
#
#
# * Change the parameters $h_1$ and $h_2$ in the above panel and observe the difference between the new graph and old graph. 
# *  What happens if you increase the $v_1$ to $v_2$. What does it tell you about the refracting material?
# *  Calculate $h_1$ and $h_2$ using the following formulae and observations made from the graph. 
#
# $$h_1 = \frac{v_1{t_i^-}}{2\cos{i_c}}$$
#
# $$h_2 = \frac{v_1{t_i^+}}{2\cos{i_c}}$$
#
# where $t_i^+$ and $t_i^-$ are the intercept times. 
#


