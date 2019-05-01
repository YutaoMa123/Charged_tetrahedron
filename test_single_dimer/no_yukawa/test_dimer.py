from hoomd import *
from hoomd import md
from hoomd import deprecated
import numpy as  np
from math import pi
import sys
import os
from PatchyTetrahedron import PatchyTetrahedron

def MIXS(s1,s2):
	return (s1+s2)/2
def MIXE(eps1,eps2):
	return np.sqrt(eps1*eps2)

context.initialize()
L = 100.0
uc = lattice.unitcell(N=2,a1 = [L,0.,0.], a2 = [0,L,0.], a3 = [0.,0.,L], position = [[L/6,0,0],[0.,0.,0.]], type_name = ['C','C'])
system = init.create_lattice(unitcell=uc,n=1)
system.particles.types.add('A')
system.particles.types.add('B')
snap = system.take_snapshot()
print (snap.particles.types)

sigma = [0.0,8.0,1.0]
types = ['C','A','B']
mass = [100.0,100.0,1.0]
n_layers = 2

tetra = PatchyTetrahedron()
tetra.setup(types,sigma,mass,n_layers)
locations = tetra.get_pos()
labels = tetra.get_labels()
type_nums = tetra.get_type_nums()

rigid = md.constrain.rigid()
rigid.set_param(types[0],types=labels,positions=locations)
rigid.create_bodies()
for p in system.particles:
	p.diameter = sigma[p.typeid]
	p.mass = mass[p.typeid]

nl = md.nlist.cell()
yukawa = md.pair.yukawa(r_cut = 24.0, nlist=nl)
for i in range(len(types)):
	yukawa.pair_coeff.set('C',types[i],epsilon=0.0,kappa=1.0)
yukawa.pair_coeff.set('A','A',epsilon = 50.0, kappa=1./24)
yukawa.pair_coeff.set('A','B',epsilon = 0.0,kappa=1./25)
yukawa.pair_coeff.set('B','B',epsilon = 0.0, kappa= 1./25)
yukawa.set_params(mode='shift')


lj = md.pair.lj(r_cut = 5,nlist =nl)
N_B_per_vertex = type_nums[2]/4
eps_B = 180.0/N_B_per_vertex
eps_A = 1.0
for i in range(len(types)):
	lj.pair_coeff.set('C',types[i],epsilon=0.0,sigma=1.0)
lj.pair_coeff.set('A','A',epsilon = 0.0, sigma= 1.0)
lj.pair_coeff.set('A','B',epsilon= 0.0, sigma = 1.0)
lj.pair_coeff.set('B','B',epsilon =eps_B, sigma=sigma[-1],r_cut= 5.0)
lj.set_params(mode='shift')

wca = md.pair.slj(r_cut=2.5,nlist=nl,d_max = max(sigma))
for i in range(len(types)):
	wca.pair_coeff.set('C',types[i],epsilon=0.0,sigma=1.0)
wca.pair_coeff.set('A','A',epsilon = eps_A, sigma=1.0,r_cut = 2**(1./6))
wca.pair_coeff.set('A','B',epsilon = MIXE(eps_A,eps_B), sigma=1.0,r_cut = 2**(1./6))
wca.pair_coeff.set('B','B',epsilon=0.0,sigma=1.0)
wca.set_params(mode='shift')

tetra_centers = group.rigid_center()
for p in tetra_centers:
	p.moment_inertia = [125.,125.,125.]
	print (p.type)
	print (p.tag)

#Randomization
# soft_repulsion = md.pair.slj(r_cut = 2**(1./6), nlist = nl,d_max=max(sigma))
# soft_repulsion.pair_coeff.set(types,types,epsilon=1.0,sigma=1.0)
# soft_repulsion.set_params(mode='shift')
# yukawa.disable()
# lj.disable()
# wca.disable()
# md.integrate.mode_standard(dt = 0.005)
# langevin1 = md.integrate.langevin(group = tetra_centers,seed = 12345 ,kT = 2.0, dscale=1.0)
# dumper = dump.gsd(filename = 'randomization.gsd', period = 1e3, group=group.all(), overwrite=True)
# logger = analyze.log(filename = 'randomization.txt',quantities=['num_particles','potential_energy','kinetic_energy','temperature'] ,period = 1e2, overwrite=True)
# run(1e6)
# dumper.disable()
# logger.disable()
# langevin1.disable()
# soft_repulsion.disable()
run_number = int(sys.argv[1])
out_dir = 'Run%d'%(run_number)
os.mkdir(out_dir)
np.random.seed()
seed = np.random.randint(1,10000)

deprecated.dump.xml(group=group.all(),filename='%s/init.xml'%(out_dir),all=True)
md.integrate.mode_standard(dt = 0.005)
kT = variant.linear_interp(points = [(0, 1.0), (5e6, 0.0)])
langevin1 = md.integrate.langevin(group = tetra_centers,seed = seed ,kT = 1.0, dscale=1.0)
#nve = md.integrate.nve(group=tetra_centers)
yukawa.disable()
dumper = dump.gsd(filename = '%s/test_production_no_yukawa.gsd'%(out_dir), period = 1e3, group=group.all(), overwrite=True)
logger = analyze.log(filename = '%s/test_production_no_yukawa.txt'%(out_dir),quantities=['num_particles','potential_energy','kinetic_energy','temperature'] ,period = 1e4, overwrite=True)
run(5e6)
dumper.disable()
logger.disable()

