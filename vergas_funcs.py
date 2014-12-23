# A library for the molecular dynamics simulation of a gas using verlet integration
# William Gilpin 2014

from matplotlib import pyplot
from scipy import *
from numpy import *
from random import randrange

# NOTE: it might be more clever to construct entire simulation as a CLASS with dt, dx, dy as attributes
# and an explicit generator/yield function for each new timestep.

# def V(r, eps=100.0, sig=1.0e-1):
def V(r, eps=100.0, sig=1.0e-2):
    # Lennard-Jones potential
    eout = eps*((sig/r)**12 - (sig/r)**6)
#     eout[isnan(eout)] = 1e1*rand()
    eout[r==0.0] = 0.0
    return eout

def init_lattice(N, L=1.0, d=2):
    # Build an NxN lattice of real size L in which the average kinetic energy of 
    # the starting particles is set by the temperature T
    num_pdim = ceil(N**(1./d))
    x = tile(linspace(L/(2.*num_pdim),L-L/(2.*num_pdim), num_pdim), num_pdim)
    y = kron(linspace(L/(2.*num_pdim),L-L/(2.*num_pdim), num_pdim), ones(num_pdim))
            
    return vstack((x,y))


def period_bc(coords, L=1.0):
    # boundary copying is effective up to 1/2 the box diameter, which is
    # for the the short-ranged LJ potential
    push0 = zeros(coords.shape)
    push0[0, :] = L
    push1 = zeros(coords.shape)
    push1[1, :] = L
    coords1 = hstack((coords, coords + push0, coords - push0, \
                      coords + push1, coords - push1, coords + push0 + push1,\
                      coords + push0 - push1, coords - push0 + push1, coords - push0 - push1))
    return coords1

def tile_stack(tle, num):
    # helpful function for doing kronecker product of a row vector
    # and a column vector
    return reshape( tile(tle, num), (len(tle), num) )

def update_pos(coords, prev_coords, dt=1e-2, dx=1e-3, dy=1e-3,  L=1.0, m=1.0):
    # Use first-order verlet to update the position of a list of particle given snapshots
    # of their coordinates at two consecutive timesteps
    
    new_coords = list()
    new_vels = list()
    
    coords1 = period_bc(coords)
    acc = zeros(coords.shape)

    for pair in zip(coords.T, prev_coords.T):
        
        coord = pair[0]
        prev_coord = pair[1]

        r0 = sqrt(sum(( coords1 - tile_stack(coord, len(coords1.T)) )**2, axis=0))
        
        r1px = sqrt(sum(( coords1 - tile_stack(coord + array([dx, 0]), len(coords1.T)) )**2, axis=0))
        r1nx = sqrt(sum(( coords1 - tile_stack(coord - array([dx, 0]), len(coords1.T)) )**2, axis=0))
        r1py = sqrt(sum(( coords1 - tile_stack(coord + array([0, dy]), len(coords1.T)) )**2, axis=0))
        r1ny = sqrt(sum(( coords1 - tile_stack(coord - array([0, dy]), len(coords1.T)) )**2, axis=0))
        
        acc = (-1./m)*array([ (V(r1px) - V(r1nx))/(2.*dx) , (V(r1py) - V(r1ny))/(2.*dy) ])
        acc = sum(acc, axis=1)

        new_coord = 2*coord - prev_coord + acc*(dt**2)     
        new_coord = mod(new_coord, L)

        new_coords.append(new_coord)
        
    new_coords = array(new_coords)
        
    return new_coords.T


def rand_vels(numb, T):
    # returns a random vector of length numb such that
    # the sum of squares adds up to T
    
    T = double(T)
    vec = ones(numb)*T
    pert = .3*vec[0]

    for ii in xrange(8*numb):

        ri2 = randrange(0, len(vec))
        if ((vec[ri2] - pert) > 0.0):
            vec[ri2] = vec[ri2] - pert
            ri1 = randrange(0, len(vec))
            vec[ri1] = vec[ri1] + pert
        else:
            pass
        
    return sqrt(vec)

def rand_vcomps(vels):
    # takes a list of vector lengths and returns their components randomly
    # projected along two axes
    
    comps = list()
    for vel in vels:
        theta = 2*pi*rand()
        comps.append((vel*cos(theta), vel*sin(theta)))
    return array(comps)

def verlet_gas(coords0, T=10.0, nsteps=100, dt=1e-2, L=1.0):
    # wrapper for verlet routine that repeatedly calls update_pos.py to increment the positions
    # of a list of particles initialized to a given temperature
    
    time = linspace(0.0, nsteps*dt, nsteps)
    all_coords = list()
    
    vels0 = rand_vels(len(coords0.T), T)
    vcomps0 = rand_vcomps(vels0).T
    prev_coords = coords0 - dt*vcomps0
    prev_coords = mod(prev_coords, L)
    
    all_coords.append(prev_coords)
    all_coords.append(coords0)
    
    for ii in xrange(nsteps):
        nxt = update_pos(all_coords[-1:][0], all_coords[-2:-1][0], dt=dt)
        all_coords.append(nxt)
        
    return all_coords

def auto_corr(all_coords, ind=0):
    # Calculate the cumulative autocorrellation function for a particle
    # specified by the index 'ind'
    r0 = all_coords[0][:,ind]
    corr = list()
    # might need to think about wraparound here
    for coords in all_coords:
        r = coords[:,ind]
        corr.append(sum((r - r0)**2))
    corr = array(corr)
    
    return cumsum( corr/(arange(len(corr))+1.) )