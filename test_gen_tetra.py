from hoomd import *
from hoomd import md
from hoomd import deprecated
import numpy as  np
from math import pi
from PatchyTetrahedron import PatchyTetrahedron

def MIXS(s1,s2):
	return (s1+s2)/2
def MIXE(eps1,eps2):
	return np.sqrt(eps1*eps2)

context.initialize()
L = 35.0
uc = lattice.unitcell(N=2,a1 = [L,0.,0.], a2 = [0,L,0.], a3 = [0.,0.,L], position = [[L/2,0,0],[0.,0.,0.]], type_name = ['C','C'])
sys = init.create_lattice(unitcell=uc,n=3)
sys.particles.types.add('A')
sys.particles.types.add('B')
snap = sys.take_snapshot()
print (snap.particles.types)

sigma = [1.0,0.0,8.0,1.0]
types = ['D','C','A','B']
mass = [1.0,600.0,125.0,1.0]
n_layers = 3

tetra = PatchyTetrahedron()
tetra.setup(types[1:],sigma[1:],mass[1:],n_layers)
locations = tetra.get_pos()
labels = tetra.get_labels()
type_nums = tetra.get_type_nums()

rigid = md.constrain.rigid()
rigid.set_param(types[1],types=labels,positions=locations)
rigid.create_bodies()
for p in sys.particles:
	p.diameter = sigma[p.typeid]
	p.mass = mass[p.typeid]


# Set up pair potentials
nl = md.nlist.cell()
# dlvo = md.pair.DLVO(r_cut = 2.5, nlist = nl)
# dlvo.pair_coeff.set('C',types,kappa=1.0,Z=0.0,A=0.0)
# dlvo.pair_coeff.set('D','D',kappa=1.0,Z=0.0,A=0.0)
# dlvo.pair_coeff.set('D','A',kappa=1.0,Z=0.0,A=-6.0)
# dlvo.pair_coeff.set('D','B',kappa=1.0,Z=0.0,A=-6.0)
# dlvo.pair_coeff.set('A','A',kappa=1.0,Z=0.0,A=0.0)
# dlvo.pair_coeff.set('A','B',kappa=1.0,Z=0.0,A=0.0)
# dlvo.pair_coeff.set('B','B',kappa=1.0,Z=0.0,A=0.0)
# dlvo.set_params(mode="shift")

yukawa = md.pair.yukawa(r_cut = 25.0, nlist=nl)
for i in range(len(types)):
	yukawa.pair_coeff.set('C',types[i],epsilon=0.0,kappa=1.0)
yukawa.pair_coeff.set('D','D',epsilon = 64.0,kappa=1./5,r_cut = 5.0)
yukawa.pair_coeff.set('D','A',epsilon = -16.0,kappa=1./5,r_cut=5.0)
yukawa.pair_coeff.set('D','B',epsilon = 0.0, kappa=1./25)
yukawa.pair_coeff.set('A','A',epsilon = 10.0, kappa=1./25)
yukawa.pair_coeff.set('A','B',epsilon = 0.0,kappa=1./25)
yukawa.pair_coeff.set('B','B',epsilon = 0.0, kappa= 1./25)
yukawa.set_params(mode='shift')


lj = md.pair.lj(r_cut = 2.5,nlist =nl)
N_B_per_vertex = type_nums[2]/4
eps_B = 20.0/N_B_per_vertex
eps_A = 2.0
for i in range(len(types)):
	lj.pair_coeff.set('C',types[i],epsilon=0.0,sigma=1.0)
lj.pair_coeff.set('D','D',epsilon = 0.0,sigma = 1.0)
lj.pair_coeff.set('D','A',epsilon = 0.0, sigma = 1.0)
lj.pair_coeff.set('D','B',epsilon = 0.0, sigma = 1.0)
lj.pair_coeff.set('A','A',epsilon = 0.0, sigma= 1.0)
lj.pair_coeff.set('A','B',epsilon= 0.0, sigma = 1.0)
lj.pair_coeff.set('B','B',epsilon =eps_B, sigma=sigma[3],r_cut=2.5)
lj.set_params(mode='shift')

wca = md.pair.slj(r_cut=2.5,nlist=nl,d_max = max(sigma))
for i in range(len(types)):
	wca.pair_coeff.set('C',types[i],epsilon=0.0,sigma=1.0)
wca.pair_coeff.set('D','D',epsilon=1.,sigma=1.0,r_cut=2**(1./6))
wca.pair_coeff.set('D','A',epsilon=1.,sigma=1.0,r_cut=2**(1./6))
wca.pair_coeff.set('D','B',epsilon=1.,sigma=1.0,r_cut=2**(1./6))
wca.pair_coeff.set('A','A',epsilon = eps_A, sigma=1.0,r_cut = 2**(1./6))
wca.pair_coeff.set('A','B',epsilon = MIXE(eps_A,eps_B), sigma=1.0,r_cut = 2**(1./6))
wca.pair_coeff.set('B','B',epsilon=0.0,sigma=1.0)
wca.set_params(mode='shift')


# Run
deprecated.dump.xml(group=group.all(),filename='init.xml',all=True)
tetra_centers = group.rigid_center()
for p in tetra_centers:
	p.moment_inertia = [125.,125.,125.]
	print (p.type)
	print (p.tag)
ions = group.type(type='D')
apply_integrator = group.union(name='apply_integrator',a=tetra_centers,b=ions)

# Randomization
soft_repulsion = md.pair.slj(r_cut = 2**(1./6), nlist = nl,d_max=max(sigma))
soft_repulsion.pair_coeff.set(types,types,epsilon=1.0,sigma=1.0)
soft_repulsion.set_params(mode='shift')
yukawa.disable()
lj.disable()
wca.disable()
md.integrate.mode_standard(dt = 0.005)
langevin1 = md.integrate.langevin(group = apply_integrator,seed = 12345 ,kT = 1.0, dscale=1.0)
dumper = dump.gsd(filename = 'randomization.gsd', period = 1e3, group=group.all(), overwrite=True)
logger = analyze.log(filename = 'randomization.txt',quantities=['num_particles','potential_energy','kinetic_energy','temperature'] ,period = 1e2, overwrite=True)
run(1e6)
dumper.disable()
logger.disable()
langevin1.disable()
soft_repulsion.disable()

# Test production run
langevin2 = md.integrate.langevin(group = apply_integrator,seed = 23456 ,kT = 0.8, dscale=1.0)
yukawa.enable()
lj.enable()
wca.enable()
dumper = dump.gsd(filename = 'test_production.gsd', period = 1e3, group=group.all(), overwrite=True)
logger = analyze.log(filename = 'test_production.txt',quantities=['num_particles','potential_energy','kinetic_energy','temperature'] ,period = 1e4, overwrite=True)
run(1e6)
dumper.disable()
logger.disable()