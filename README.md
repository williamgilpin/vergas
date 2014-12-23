## Description

The function simlj.py conducts a simulation of a Lennard-Jones gas with a random initial distirbution of particle energies set by the temperature parameter, T. At each timestep, a snapshot of the positions and momenta of all particles is recorded, and, on hte simulation is complete, these snapshots are used to compute metrics like the autocorrelation of particles within the gas, which provides an estimate of hte diffusivity at different temperatures. The non-monotonicity of the dependence of hte diffusivity on temperature is a characteristic of the Lennard-Jones potential, which is attractive over short distances and repulsive over large distances.

The library vergas_funcs.py is the library of functions necessary to conduct the simulation