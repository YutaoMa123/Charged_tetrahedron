from hoomd import *
from hoomd import md
from hoomd import deprecated
import numpy as  np
from math import pi
from PatchyRings import PatchyRings

R = 2.5
r = 0.25
N_patchy_layers = 5
alpha = np.arccos(1-(2*r)**2/(2*R**2))
phi_list = [0.]
for i in range(N_patchy_layers):
	phi_list.append(i*alpha)
type_list = ['A']
for i in range(N_patchy_layers):
	type_list.append('B')
sigma_list = [2*R]
for i in range(N_patchy_layers):
	sigma_list.append(2*r)
mass_list = [125.]
for i in range(N_patchy_layers):
	mass_list.append(1.0)

context.initialize()
sys = init.create_lattice(unitcell=lattice.sc(a=13.0), n = 1)
sys.particles.types.add('B')

ring = PatchyRings()
ring.setup(types=type_list,sigma=sigma_list,angles = phi_list, mass = mass_list)
type_nums = ring.getTypeNums()
print (len(ring.getTypes()))
print (len(ring.getPos()))
rigid = md.constrain.rigid()
rigid.set_param(type_list[0],types = ring.getTypes(), positions = ring.getPos())
rigid.create_bodies()
for p in sys.particles:
	p.diameter = sigma_list[p.typeid]
	p.mass = mass_list[p.typeid]

deprecated.dump.xml(group=group.all(),filename='test_ring_with_rotation.xml',all=True)



