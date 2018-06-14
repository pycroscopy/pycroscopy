
This folder contains code for the simulation of an atomic force microscopy probe (parabolic) penetrating a viscoelastic surface containing multiple characteristic times.
The contact mechanics of:

- (Lee, E. Ho, and Jens Rainer Maria Radok. "The contact problem for viscoelastic bodies."
 Journal of Applied Mechanics 27, no. 3 (1960): 438-444.)


Folder structure:
The repository contains the following files:

- AFM_lib.py  --> This is the library containing the main functions to make the simulations.
- AFM_calculations  --> This library contains functions to perform postprocessing of simulations data such as calculation of amplitude and phase of the tip trajectory.
- Simulation_SoftMatter.ipynb  --> This jupyter notebook performs the simulations for single tapping mode with the 1st mode excited
- PIB.txt -- > contains the values of the Generalized Maxwell parameters for the case of polyisobutylene sample (left column relaxation times, right column moduli of each arm). This data for
polyisobutylene has been extracted from: "Brinson, Hal F., and L. Catherine Brinson. "Polymer engineering science and viscoelasticity." An Introduction (2008)."


These simulations have been used in:
"Nikfarjam, M., López-Guerra, E. A., Solares, S. D., & Eslami, B. (2018). Imaging of viscoelastic soft 
matter with small indentation using higher eigenmodes in single-eigenmode amplitude-modulation atomic force microscopy. 
Beilstein journal of nanotechnology, 9, 1116."



