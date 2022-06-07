


See the PDF file to have the complete report.



# **Introduction**

The building sector is an energy consuming sector. It will then be very important to succeed in reducing the energy consumed by the buildings and to do this there are several parameters to take into account such as the geometry of the building and the materials which constitute it. The study made during these three sessions consists in the estimation of the energy efficiency of a smart home by modeling and simulation.

The plan for the three sessions can be found below:

Session 1 - **Model**

Session 2 - **Implementation** (Python code + run/debug)

Session 3 - **Optimization of the code** (design/operation)


# **Hypothesis **

The heat transfer is assumed to take place in a single direction, and the heat transfer is considered only through the walls, on which the temperature is uniform at all points (heat transfer with the floor and ceiling are neglected). Also, all heat exchanges are considered linearized.


# **I_Building plan**

This study concerns a studio of approximately 28 m<sup>2</sup> composed of two rooms including a bathroom/toilet and a main room. The main room is assumed to consist of a kitchen, a dining room as well as a small room separated from the kitchen by a small wall which acts as a separation.

The studio has two windows with views to the outside, one in the bedroom and one in the dining room. Note that the walls on the left and right are adjacent to other apartments and therefore heat exchange through these walls are not considered. The main door of the apartment opens onto a corridor common to the apartments of the building. We will model the heat exchanges between the apartment and the outside. The precise dimensions of the studio are shown in figure 1 below. 



<p id="gdcalert3" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image3.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert4">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image3.jpg "image_tooltip")




<p id="gdcalert4" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image4.jpg). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert5">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image4.jpg "image_tooltip")


_Figure 1. The studied building and its dimensions with an assumed ceiling height of 2,4 meters. _


# 


# **II_Thermal and mathematical model**


## **1.Composition of the walls **

A heat flow could be described as a flow of energy due to variances in temperature between two positions, and it can occur in the form of conduction, convection, radiation as well as advection, similar to the case presented in figure 3. 



<p id="gdcalert5" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image5.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert6">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image5.png "image_tooltip")


_Figure 3. Heat exchange through a wall in the form of conduction, convection and radiation. Advection is not presented in the figure but can be described as a heat flow due to a mass transfer mechanism. _

In this studied case, the walls are considered to consist of several layers of materials influencing the heat flow and therefore the internal temperature in the building. For example, the exterior wall is considered to consist of both concrete and insulation as presented in figure 4. 



<p id="gdcalert6" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image6.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert7">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image6.png "image_tooltip")


_Figure 4. The exterior wall is considered to consist of concrete and fiberglass insulation. _



The walls of this room aren’t all the same : 



* the wall of the main-room with the kitchen to the outside (wall 1)

    & to the corridor (wall 4) : concrete + insulation

* the wall of the main-room with the kitchen to the bathroom (wall 2) : concrete + wood
* the wall of the bathroom to the corridor (wall 3): concrete + wood

All the doors are in wood, and we consider the windows to be in glass. 


## **2. Parameters**

_Table 1. Description of the materials used in the building model regarding thickness, heat transfer coefficient, density as well as specific heat capacity. _


<table>
  <tr>
   <td><strong>Material</strong>
   </td>
   <td><strong>Thickness [mm]</strong>
   </td>
   <td><strong>Lambda [W/m/K]</strong>
   </td>
   <td><strong>Density [kg/m<sup>3</sup>]</strong>
   </td>
   <td><strong>Specific heat capacity [J/kgK] </strong>
   </td>
  </tr>
  <tr>
   <td>Air
   </td>
   <td>-
   </td>
   <td>0,0244
   </td>
   <td>1,208
   </td>
   <td>1006
   </td>
  </tr>
  <tr>
   <td>Concrete
   </td>
   <td>130
   </td>
   <td>1,8
   </td>
   <td>2300
   </td>
   <td>880
   </td>
  </tr>
  <tr>
   <td>Insulation 
   </td>
   <td>100
   </td>
   <td>0,0345
   </td>
   <td>22
   </td>
   <td>850
   </td>
  </tr>
  <tr>
   <td>Glass
   </td>
   <td>20
   </td>
   <td>1,4
   </td>
   <td>2210
   </td>
   <td>730
   </td>
  </tr>
  <tr>
   <td>Wood
   </td>
   <td>20
   </td>
   <td>0,15
   </td>
   <td>532
   </td>
   <td>2700
   </td>
  </tr>
</table>


_Table 2. Description of the surfaces regarding the building model components. _


<table>
  <tr>
   <td><strong>Part of building</strong>
   </td>
   <td><strong>Surface [m<sup>2</sup>]</strong>
<p>
<strong>We chose a height of 2,40m</strong>
   </td>
  </tr>
  <tr>
   <td>Wall 1 (Exterior wall)
   </td>
   <td>13,6
   </td>
  </tr>
  <tr>
   <td>Wall 2 (Bathroom outer walls)
   </td>
   <td>14,9
   </td>
  </tr>
  <tr>
   <td>Wall 3 (Hallway to corridor) 
   </td>
   <td>5,7
   </td>
  </tr>
  <tr>
   <td>Wall 4 (Bathroom to corridor)
   </td>
   <td>10,6
   </td>
  </tr>
  <tr>
   <td>Windows 
   </td>
   <td>3
   </td>
  </tr>
  <tr>
   <td>Door
   </td>
   <td>2,1
   </td>
  </tr>
</table>



## **3.Thermal model of the room**

Below, in figure 5, the thermal modeling of the room is presented. The model is based on the heat flows which take place in the building plan described in figure 1. Further, the model consists of different types of components including branches, nodes and external heat sources representing how the heat flows through, for example, walls and windows.  

 

<p id="gdcalert7" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image7.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert8">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image7.png "image_tooltip")


_Figure 5. The thermal model of the studied system based on the presented model in figure 1. The nodes and resistances were numbered appropriately and utilized to determine the matrices for the model to be adopted. _

Inner nodes of each wall material and each room has the following** heat capacity**:

**C = ρ. c. e. S.**

where:



* ρ is the density of the different materials ( kg/m3)
* c is the specific heat mass (J/kg/K)
* S is the surface of the wall (m2)
* e is the thickness of the wall (m)

For the **convective resistances**, they could be described by the formula

**R =1/(h.S)**

where:



* h is the thermal convection coefficient (W/m2/K) 

The values of **conduction resistance** are** R =e/(λ.S)**

where:



* λ is the thermal conductivity of the wall part (W/m/K)


## **4. Code implementation **

The construction of the thermal model requires the different principles of thermodynamics:



<p id="gdcalert8" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image8.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert9">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image8.png "image_tooltip")


We have the following three equations:



<p id="gdcalert9" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image9.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert10">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image9.png "image_tooltip")


where e is the vector of temperature differences between two nodes, and

where q is the vector of heat flow through each resistor.

The three equations above combined lead us to the following differential equation, which governs the system :



<p id="gdcalert10" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image10.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert11">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image10.png "image_tooltip")



## 


## **5. Matrix A, C and G **

For our model, we have 28 temperature nodes and 33 heat flows (we consider the controller Kp in it) . Therefore, the matrices and the vectors that will be created will depend on this number of nodes and flows.


    We have then:



* the matrix **A** (33x28) which relates the temperatures between the nodes (composed of 1 for incoming flows to the node, -1 for an outgoing flow, and the rest is made of 0).
* the vector **b** (33x1) which relates the temperature sources
* the vector **f** (28x1), the vector of the flux sources
* the matrix **C** (28x28) for the thermal capacities
* the **G** matrix (33x33) for the conductances
* finally, the vector **Ө** (28x1) which concerns the temperatures at each node

**About the matrix A : **

(The matrix A is very big so here is a reconstruction of what we obtained with the code)

A = zeros (33,28)



<p id="gdcalert11" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image11.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert12">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image11.png "image_tooltip")


**About the G matrix : **

You can see in **Annexe 2** the exact calculations, so the exact components of the matrix wrote down : 

We used : 

<p id="gdcalert12" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: equation: use MathJax/LaTeX if your publishing platform supports it. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert13">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

 = 

<p id="gdcalert13" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: equation: use MathJax/LaTeX if your publishing platform supports it. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert14">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

 ,  

<p id="gdcalert14" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: equation: use MathJax/LaTeX if your publishing platform supports it. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert15">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

=  

<p id="gdcalert15" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: equation: use MathJax/LaTeX if your publishing platform supports it. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert16">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

 = h * S

Where λ is the conductivity of the material (as defined in Table 1), S is the surface through which the heat flux is transmitted, w is the thickness of the wall layer (or slice of it), h is the convection coefficient or the equivalent radiative coefficient

G = diag (g)

with g :** **

<p id="gdcalert16" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image12.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert17">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image12.png "image_tooltip")




<p id="gdcalert17" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image13.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert18">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image13.png "image_tooltip")


**About the matrix C : **

You can see in **Annexe 3** the exact calculations, so the exact components of the matrix wrote down : 

As said before, we used the formula : **C = ρ. c. e. S.**

The C matrix is very big.

So here is only the c matrix, with C =diag(c)



<p id="gdcalert18" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image14.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert19">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image14.png "image_tooltip")


For the simulation, we erased all the low conductances. 


## **6. Vectors b and f in time **

The vector of temperature sources is b. It has the size of the number of branches and if its element is corresponding to a branch without a source is zero. If the flow in a source is from low potential to high potential (i.e. from - to +), the source is positive

Here we have constraints on the model which are temperatures sources : 



<p id="gdcalert19" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image15.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert20">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image15.png "image_tooltip")


The vector **f** (28x1), the vector of the flux sources : 

f[1,1] = 1 (corresponds to the** outdoor** radiative flow rate)

f[5,1] = 1 (corresponds to the **indoor** radiative flow rate)

f[6,1] = 1 (corresponds to the **indoor** radiative flow rate)

f[13,1] = 1 (corresponds to the **absorbed** radiative flow rate)

f[14,1] = 1 (corresponds to the** indoor** radiative flow rate)

Regarding the radiation : 

<p id="gdcalert20" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: equation: use MathJax/LaTeX if your publishing platform supports it. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert21">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

 = hr (Ts - Te)

We will use : 1000 W/m2 K and 500 W/m2 K values for our simulation. 



<p id="gdcalert21" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image16.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert22">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image16.png "image_tooltip")



# 


# **III_Simulation **


## **1.Steady-state **

We will use :  h<sub>out</sub> = air convection coefficient = 25 & h<sub>in</sub> = 8 

For all the simulations we perform, it is necessary to go through a state representation of the system. The transition from thermal matrix modelling to the state representation corresponds to the transition between this equation :



<p id="gdcalert22" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image17.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert23">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image17.png "image_tooltip")


and the state space model : 

<p id="gdcalert23" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image18.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert24">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image18.png "image_tooltip")


where:

𝐱 - the state vector: temperature nodes with capacities;

𝐲 - output vector: temperatures in the nodes of interest;

𝐮 - input vector: temperature and flow-rate sources;

𝐀𝑠 - state matrix;

𝐁𝑠 - input matrix;

𝐂𝑠 - output matrix;

𝐃𝑠 - feedthrough matrix.

The input vector u is : 

<p id="gdcalert24" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image19.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert25">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image19.png "image_tooltip")


This transformation is achieved with the fTC2SS function by coding as follows : [As,Bs,Cs,Ds] = fTC2SS(A,G,b,C,f,y)

For this simulation we consider an initial temperature of 20°C in the main room, an outside temperature of 5°C. In addition, we consider that there is no heating system in operation and that the solar radiant flux is zero.

We obtain the following values : 



<p id="gdcalert25" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image20.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert26">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image20.png "image_tooltip")



## **2. Steady -state : step response **

We simulate a step response using the following vector: 

b[[0, 15, 32]] = 0 + np.array([1, 1, 20]

It means that the outside temperature is 1°C, the initial temperature of the main room is 20°C and we use a controller to have the temperature of the main room to 1°C. 

The temperature shown is from the main room. It goes from 20°C to 1°C. 


# **3 _ Dynamic model **

In dynamic simulation, the inputs vary in time. Let's indicate the sources in the circuit by ones.

Let's consider that the output of the circuit is the indoor temperature. 

The state-space representation is obtained from the differential-algebraic equations of the thermal circuit :

[As, Bs, Cs, Ds] = dm4bem.tc2ss(A, G, b, C, f, y)

The maximum time step for numerical stability of Euler explicit integration in time is given by the minimum eigenvalue 

<p id="gdcalert26" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: equation: use MathJax/LaTeX if your publishing platform supports it. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert27">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

 of the state matrix 

<p id="gdcalert27" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: equation: use MathJax/LaTeX if your publishing platform supports it. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert28">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>

 : 



<p id="gdcalert28" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image21.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert29">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image21.png "image_tooltip")


The maximum time step for numerical stability of Euler explicit integration in time depends on the value of the P-controller gain and if the capacities of the indoor air and window glass are taken into account.

**We obtain a maximum time step of 30,61 sec. **

**Let’s look at the step response : **

For the input vector u, the outdoor temperatures will be T<sub>o</sub> = 1, the indoor set-point temperature will be T<sub>sp</sub> = 0, and the heat flow sources will be zero.

The vectors for the evolution of temperatures (states) in time for Euler explicit and implicit are:

temp_exp **=** np**.**zeros([n_tC, t**.**shape[0]])

temp_imp **=** np**.**zeros([n_tC, t**.**shape[0]])

By integrating the state space model : 



<p id="gdcalert29" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image22.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert30">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image22.png "image_tooltip")


by using Euler forward (or explicit) : 



<p id="gdcalert30" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image23.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert31">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image23.png "image_tooltip")


and Euler backward (or implicit) integration : 



<p id="gdcalert31" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image24.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert32">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image24.png "image_tooltip")


we obtain the outputs.

**The obtained curves are presented at the following page:  **



<p id="gdcalert32" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image25.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert33">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image25.png "image_tooltip")


<p id="gdcalert33" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image26.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert34">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image26.png "image_tooltip")




<p id="gdcalert34" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/image27.png). Store image on your image server and adjust path/filename/extension if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert35">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/image27.png "image_tooltip")



## **4.Simulation with weather data **

We didn’t succeed at simulating this part. 

But we could have seen the impact of radiation on the indoor temperature. 


# **Conclusion **

It is crucial to analyse the heat flow in a building to be able to enable energy management and optimise the energy consumption of the building. By creating a model of a building and considering heat flows in the form of conduction, convection and radiation in both steady state and dynamic conditions, this can be accomplished. In this project, we were able to develop a coherent thermal model of a building and verify the relevance of our modelling choices through simulations. The obtained results are consistent for both the static and the dynamic model, but it can be seen that the obtained time step of about 30 seconds regarding the dynamic model, is relatively small. This implies that the simulation possibly could be improved. 

Further, a controller was implemented which modelised a HVAC model capable of heating and cooling. Regarding this, it could be seen that the time step depends on Kp affecting the performance of the controller. When Kp goes towards the infinity, the controller becomes more idealised and when the value of Kp goes towards zero, the controller becomes less efficient. 


# 


# References 

L. Pérez-Lombard, J. Ortiz, C. Pout (2007). _A review on buildings energy consumption information_. Energy and Buildings, Vol. 40, Iss. 3, p. 394-398. Available at: [https://www.sciencedirect.com/science/article/pii/S0378778807001016](https://www.sciencedirect.com/science/article/pii/S0378778807001016) [2022-04-22] 



<span style="text-decoration:underline;">Annexe 1 : Precise table about the composition of the walls</span>


<table>
  <tr>
   <td><strong>Material</strong>
   </td>
   <td><strong>Thickness [mm]</strong>
<p>
<strong>Width (w)</strong>
   </td>
   <td><strong>Lambda [W/mK]</strong>
<p>


<p id="gdcalert35" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: equation: use MathJax/LaTeX if your publishing platform supports it. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert36">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


   </td>
   <td><strong>Density [kg/m<sup>3</sup>]</strong>
<p>


<p id="gdcalert36" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: equation: use MathJax/LaTeX if your publishing platform supports it. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert37">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


   </td>
   <td><strong>Specific heat capacity [J/kgK]</strong>
<p>
<strong>c</strong>
   </td>
   <td>Surface in m2
<p>
(height of 2,40 m )
   </td>
  </tr>
  <tr>
   <td>Air
   </td>
   <td>-
   </td>
   <td>0.0244
   </td>
   <td>1.208
   </td>
   <td>1006
   </td>
   <td>-
   </td>
  </tr>
  <tr>
   <td>Concrete 1
   </td>
   <td>130
   </td>
   <td>1.8
   </td>
   <td>2300
   </td>
   <td>880
   </td>
   <td>6.93*2.40-g=13.632
   </td>
  </tr>
  <tr>
   <td>Concrete 2
   </td>
   <td>130
   </td>
   <td>1.8
   </td>
   <td>2300
   </td>
   <td>880
   </td>
   <td>2.4*(4.43+1.78)=14.904
   </td>
  </tr>
  <tr>
   <td>Concrete 3
   </td>
   <td>130
   </td>
   <td>1.8
   </td>
   <td>2300
   </td>
   <td>880
   </td>
   <td>2.36*2.4=5.664
   </td>
  </tr>
  <tr>
   <td>Concrete 4
   </td>
   <td>130
   </td>
   <td>1.8
   </td>
   <td>2300
   </td>
   <td>880
   </td>
   <td>4.43*2.4=10.632
   </td>
  </tr>
  <tr>
   <td>Insulation 1
   </td>
   <td>100
   </td>
   <td>0.0345
   </td>
   <td>22
   </td>
   <td>850
   </td>
   <td>6.93*2.4-g=13.632
   </td>
  </tr>
  <tr>
   <td>Insulation 3
   </td>
   <td>100
   </td>
   <td>0.0345
   </td>
   <td>22
   </td>
   <td>850
   </td>
   <td>2.36*2.4=5.664
   </td>
  </tr>
  <tr>
   <td>Insulation 4
   </td>
   <td>100
   </td>
   <td>0.0345
   </td>
   <td>22
   </td>
   <td>850
   </td>
   <td>4.43*2.4=10.632
   </td>
  </tr>
  <tr>
   <td>Glass 
   </td>
   <td>20 
   </td>
   <td>1.4
   </td>
   <td>2210
   </td>
   <td>730
   </td>
   <td>2*1.5=3
   </td>
  </tr>
  <tr>
   <td>Wood 2
   </td>
   <td>20
   </td>
   <td>0.15
   </td>
   <td>532
   </td>
   <td>2700
   </td>
   <td>2.4*(4.43+1.78)=14.904
   </td>
  </tr>
  <tr>
   <td>Wood door bathroom
   </td>
   <td>20
   </td>
   <td>0.15
   </td>
   <td>532
   </td>
   <td>2700
   </td>
   <td>2.10*1=2.1
   </td>
  </tr>
  <tr>
   <td>Wood door corridor 
   </td>
   <td>20
   </td>
   <td>0.15
   </td>
   <td>532
   </td>
   <td>2700
   </td>
   <td>2.10*1=2.1
   </td>
  </tr>
</table>




<span style="text-decoration:underline;">Annexe 2 : Construction of the g matrix</span>


<table>
  <tr>
   <td>About the construction of the matrix g
   </td>
  </tr>
  <tr>
   <td>g=zeros(32,1)
<p>
Exterior wall : 
<p>
g(1,1) = ho * S_concrete1
<p>
g(2,1) = 2*_concrete * S_concrete1 / w_concrete
<p>
g(3,1) = 2*_concrete * S_concrete1 / w_concrete
<p>
g(4,1) = 2*_insulation * S_insulation1 / w_insulation
<p>
g(5,1) = 2*_insulation * S_insulation1 / w_insulation
<p>
g(6,1) = hin * S_insulation1
<p>
Wall main door - bathroom : 
<p>
g(7,1) = hin * S_wood2
<p>
g(8,1) = 2*_wood * S_wood2 / w_wood
<p>
g(9,1) = 2*_wood * S_wood2 / w_wood
<p>
g(10,1) = 2*_concrete * S_concrete2 / w_concrete
<p>
g(11,1) = 2*_concrete * S_concrete2 / w_concrete
<p>
g(12,1) = hin * S_concrete2
<p>
Door bathroom : 
<p>
g(13,1) = hin * S_doorbathroom
<p>
g(14,1) = 2*_wood * S_doorbathroom / w_wood
<p>
g(15,1) = hin * S_doorbathroom
<p>
Windows : 
<p>
g(16,1) = hout * S_windows
<p>
g(17,1) = hint * S_windows
<p>
Wall main room - corridor : 
<p>
g(18,1) = hin * S_insulation3
<p>
g(19,1) = 2*_insulation * S_insulation3 / w_insulation
<p>
g(20,1) = 2*_insulation * S_insulation3 / w_insulation
<p>
g(21,1) = 2*_concrete * S_concrete3 / w_concrete
<p>
g(22,1) = 2*_concrete * S_concrete3 / w_concrete
<p>
g(23,1) = hin * S_concrete3
<p>
Door main room - corridor : 
<p>
g(24,1) = hin * S_doorcorridor
<p>
g(25,1) = 2*_wood * S_doorcorridor / w_wood
<p>
g(26,1) = hin * S_doorcorridor
<p>
Wall bathroom - corridor : 
<p>
g(27,1) = hin * S_insulation4
<p>
g(28,1) = 2*_insulation * S_insulation4 / w_insulation
<p>
g(29,1) = 2*_insulation * S_insulation4 / w_insulation
<p>
g(30,1) = 2*_concrete * S_concrete4 / w_concrete
<p>
g(31,1) = 2*_concrete * S_concrete4 / w_concrete
<p>
g(32,1) = hin * S_concrete4
   </td>
  </tr>
</table>




<span style="text-decoration:underline;">Annexe 3 : Construction of the c matrix</span>


<table>
  <tr>
   <td>About the construction of the matrix c
   </td>
  </tr>
  <tr>
   <td>c = zeros(28,1)
<p>
Exterior wall : (concrete and insulation)
<p>
S_concrete1 =S_insulation1
<p>
c(2,1) = c_concrete * S_concrete1 * w_concrete * _concrete
<p>
c(4,1) = c_insulation * S_insulation1 * w_insulation * _insulation 
<p>
Mur principal room to bathroom : 
<p>
S_concrete2 = S_wood2
<p>
c(7,1) = c_concrete * S_concrete2 * w_concrete * _concrete
<p>
c(10,1) = c_wood * S_wood2 * w_wood *  _wood
<p>
Mur principal room to corridor : 
<p>
S_concrete3 =S_insulation3
<p>
c(15,1) = c_insulation * S_insulation3 * w_insulation * _insulation
<p>
c(18,1) = c_concrete * S_concrete3 * w_concrete * _concrete
<p>
Mur bathroom to corridor : 
<p>
S_concrete4 =S_insulation4
<p>
c(22,1) = c_insulation * S_insulation4 * w_insulation * _insulation
<p>
c(25,1) = c_concrete * S_concrete4 * w_concrete * _concrete
<p>
C = diag(c)
   </td>
  </tr>
</table>

