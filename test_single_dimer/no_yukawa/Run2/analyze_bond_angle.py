import numpy as np
from hoomd import *
from hoomd import md
from hoomd import deprecated
import gsd.hoomd
import matplotlib.pyplot as plt
from math import pi

def angle(v1,v2):
	theta = np.arccos( np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)),-1,1) )
	return np.rad2deg(theta)

def distance(pos1,pos2,box):
	disp = box.min_image(pos1 - pos2)
	return np.linalg.norm(disp)

def bonded(snap,i,j,box):
	type_list = snap.particles.types
	index_B = type_list.index('B')

	positions = snap.particles.position
	bodies = snap.particles.body
	type_id = snap.particles.typeid
	pos_i = positions[i,:]
	pos_j = positions[j,:]
	loc_B_i = np.where((bodies == i) & (type_id == index_B))[0]
	pos_B_i = positions[loc_B_i,:]
	loc_B_j = np.where((bodies == j) & (type_id == index_B))[0]
	pos_B_j = positions[loc_B_j,:]

	for m in range(len(pos_B_i)):
		for n in range(len(pos_B_j)):
			if (distance(pos_B_i[m,:],pos_B_j[n,:],box) < 2.5):
				return True
	return False

def bond_angle(snap,i,j,box):
	type_list = snap.particles.types
	index_B = type_list.index('B')
	index_A = type_list.index('A')

	positions = snap.particles.position
	bodies = snap.particles.body
	type_id = snap.particles.typeid
	pos_i = positions[i,:]
	pos_j = positions[j,:]

	disp_ij = box.min_image(pos_j-pos_i)
	loc_A_i = np.where( (bodies == i) & (type_id == index_A))[0]
	pos_A_i = positions[loc_A_i,:]
	min_theta_i = np.inf
	for k in range(len(pos_A_i)):
		disp = box.min_image(pos_A_i[k,:] - pos_i)
		theta = angle(disp_ij,disp)
		if (theta <= min_theta_i):
			min_theta_i = theta

	disp_ji = box.min_image(pos_i-pos_j)
	loc_A_j = np.where((bodies == j) & (type_id == index_A))[0]
	pos_A_j = positions[loc_A_j,:]
	min_theta_j = np.inf
	for k in range(len(pos_A_j)):
		disp = box.min_image(pos_A_j[k,:] - pos_j)
		theta = angle(disp_ji,disp)
		if (theta <= min_theta_j):
			min_theta_j = theta

	return max(min_theta_i,min_theta_j)

def compute_dihedral(b1,b2,b3):
	n1 = np.cross(b1,b2)
	n2 = np.cross(b2,b3)
	n1 = n1/np.linalg.norm(n1)
	n2 = n2/np.linalg.norm(n2)
	m1 = np.cross(n1,b2/np.linalg.norm(b2))
	x = np.dot(n1,n2)
	y = np.dot(m1,n2)
	return np.abs(np.arctan2(y,x))

def dihedral_angle(snap,i,j,box):
	type_list = snap.particles.types
	index_A = type_list.index('A')

	positions = snap.particles.position
	bodies = snap.particles.body
	type_id = snap.particles.typeid
	pos_i = positions[i,:]
	pos_j = positions[j,:]

	loc_A_i = np.where((bodies == i) & (type_id == index_A))[0]
	pos_A_i = positions[loc_A_i,:]
	loc_A_j = np.where((bodies == j) & (type_id == index_A))[0]
	pos_A_j = positions[loc_A_j,:]	

	counter = 0
	total_phi = 0
	disp_ij = box.min_image(pos_j-pos_i)
	disp_ji = box.min_image(pos_i-pos_j)
	for k in range(len(pos_A_i)):
		p_k = pos_A_i[k,:]
		b1 = np.asarray(box.min_image(pos_i - p_k))
		if (np.abs(angle(-b1,disp_ij)) < 8.0):
			continue
		min_phi = np.inf
		for l in range(len(pos_A_j)):
			p_l = pos_A_j[l,:]
			b3 = np.asarray(box.min_image(p_l-pos_j))
			if (np.abs(angle(b3,disp_ji)) < 8.0):
				continue
			cur_phi = compute_dihedral(b1,disp_ij,b3)
			if (cur_phi <= min_phi):
				min_phi = cur_phi
		total_phi += min_phi
		counter += 1
	return np.rad2deg(total_phi/(counter + 0.0))



context.initialize('--mode=cpu')
traj = gsd.hoomd.open('test_production_no_yukawa.gsd','rb')
theta = np.zeros(len(traj))
phi = np.zeros(len(traj))
for f in range(len(traj)):
	print ("Processing frame %d" %(f))
	snap = data.gsd_snapshot(filename='test_production_no_yukawa.gsd',frame = f)
	box = snap.box
	if (bonded(snap,0,1,box)):
		theta[f] = bond_angle(snap,0,1,box)
		phi[f] = dihedral_angle(snap,0,1,box)
	else:
		print ("Not bonded in frame %d!" %(f))
		theta[f] = np.nan
		phi[f] = np.nan
plt.figure()
plt.plot(theta)
plt.title('Bond Angle')
plt.xlabel('Frame number')
plt.ylabel('Degree')
plt.savefig('Bond_angle.png', bbox_inches='tight')

plt.figure()
plt.plot(phi)
plt.title('Dihedral Angle')
plt.xlabel('Frame Number')
plt.ylabel('Degree')
plt.savefig('Dihedral_angle.png', bbox_inches='tight')

plt.show()






