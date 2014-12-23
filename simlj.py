# This script runs a simulation of a Lennard-Jones gas and plots the resulting autocorrelation function
# averaged over many particles
# William Gilpin 2014

from matplotlib import pyplot
from scipy import *
from numpy import *
from random import randrange

from vergas_funcs import *

lat = init_lattice(100)

temps = linspace(100, 1000, 10)
sim_out = list()
for temp in temps:
    sim_out.append( array( verlet_gas(lat, T=temp, nsteps=1000) ) )

# plot lowest-temp trajectory
fig1 = figure(1)
for ii in xrange(sim_out[0].shape[2]):
    plot(sim_out[0][1:,0,ii],sim_out[0][1:,1,ii],'.')  
gca().set_color_cycle('None')
xlabel('x position')
ylabel('y position')
title('All particle trajectories for lowest temp')

# plot snapshot before and after 
fig2 = figure(2)
plot(sim_out[0][1,0,:],sim_out[0][1,1,:],'.') 
hold(True)
plot(sim_out[0][-1,0,:],sim_out[0][-1,1,:],'.')  
gca().set_color_cycle('None')
xlabel('x position')
ylabel('y position')
title('Positions before and after simulation for lowest temperature')
show()

# Plot autocorrelation function for each temperature
fig3 = figure(3)
for out in sim_out:
    plot(auto_corr(out),'.')
    hold(True)
xlabel('time')
ylabel('autocorrellation')

# Plot figure legend
fig4 = figure(4)
for temp in temps:
    semilogy(0, temp, '.', markersize=20)
title('Legend')
ylabel('temperature')
ylim([-10,1100])

show()
    