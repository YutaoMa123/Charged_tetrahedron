import numpy as np


class PatchyRings():
	Z_ANGLE = 0
	def __init__(self):
		print('Creating Patchy Ring Particle')

	def setup(self, types, sigma, angles,mass):
		# layer[0] is center particle
		n_layers = len(angles)
		self.typeNums = [0 for i in range(n_layers)]
		self.types = types
		self.sigma = sigma
		self.angles = angles
		self.mass = mass
		self.typeNums[0] = 1

		# generate first kind of patches 
		R = sigma[0]/2.0
		rp = sigma[1]/2.0
		v = self.gen_rp(angles[1], R, rp)
		self.typeNums[1] = v.shape[0]
		self.locs = v
		self.labels = [types[1] for i in range(self.typeNums[1])]
		self.inertia_tensor = np.diag([0.4*self.mass[0]*R**2,0.4*self.mass[0]*R**2,0.4*self.mass[0]*R**2])  # Inertia Tensor
		if (len(v.shape) == 1):
			self.inertia_tensor[0,0] += self.mass[1]*(v[1]**2 + v[2]**2)
			self.inertia_tensor[1,1] += self.mass[1]*(v[0]**2 + v[2]**2)
			self.inertia_tensor[2,2] += self.mass[1]*(v[0]**2 + v[1]**2)
			self.inertia_tensor[0,1] += -1*self.mass[1]*v[0]*v[1]
			self.inertia_tensor[1,2] += -1*self.mass[1]*v[1]*v[2]
			self.inertia_tensor[0,2] += -1*self.mass[1]*v[0]*v[2]
		else:
			self.inertia_tensor[0,0] += self.mass[1]*np.sum(v[:,1]**2 + v[:,2]**2)
			self.inertia_tensor[1,1] += self.mass[1]*np.sum(v[:,0]**2 + v[:,2]**2)
			self.inertia_tensor[2,2] += self.mass[1]*np.sum(v[:,0]**2 + v[:,1]**2)
			self.inertia_tensor[0,1] += -1*self.mass[1]*np.sum(np.multiply(v[:,0],v[:,1]))
			self.inertia_tensor[1,2] += -1*self.mass[1]*np.sum(np.multiply(v[:,1],v[:,2]))
			self.inertia_tensor[0,2] += -1*self.mass[1]*np.sum(np.multiply(v[:,0],v[:,2]))

		for i in range(2,n_layers):
			rp = sigma[i]/2.0
			vnew = self.gen_rp(angles[i], R, rp)
			v = self.binary_search(v,vnew)
			if len(v.shape) == 1:
				self.typeNums[i] = 1
				self.inertia_tensor[0,0] += self.mass[1]*(v[1]**2 + v[2]**2)
				self.inertia_tensor[1,1] += self.mass[1]*(v[0]**2 + v[2]**2)
				self.inertia_tensor[2,2] += self.mass[1]*(v[0]**2 + v[1]**2)
				self.inertia_tensor[0,1] += -1*self.mass[1]*v[0]*v[1]
				self.inertia_tensor[1,2] += -1*self.mass[1]*v[1]*v[2]
				self.inertia_tensor[0,2] += -1*self.mass[1]*v[0]*v[2]
			else:
				self.typeNums[i] = v.shape[0]
				self.inertia_tensor[0,0] += self.mass[1]*np.sum(v[:,1]**2 + v[:,2]**2)
				self.inertia_tensor[1,1] += self.mass[1]*np.sum(v[:,0]**2 + v[:,2]**2)
				self.inertia_tensor[2,2] += self.mass[1]*np.sum(v[:,0]**2 + v[:,1]**2)
				self.inertia_tensor[0,1] += -1*self.mass[1]*np.sum(np.multiply(v[:,0],v[:,1]))
				self.inertia_tensor[1,2] += -1*self.mass[1]*np.sum(np.multiply(v[:,1],v[:,2]))
				self.inertia_tensor[0,2] += -1*self.mass[1]*np.sum(np.multiply(v[:,0],v[:,2]))
			self.locs = np.vstack((self.locs,v))
			self.labels = np.hstack((self.labels, [types[i] for x in range(self.typeNums[i])]))
		self.inertia_tensor[1,0] = self.inertia_tensor[0,1]
		self.inertia_tensor[2,0] = self.inertia_tensor[0,2]
		self.inertia_tensor[2,1] = self.inertia_tensor[1,2]
		print('Inertia Tensor: ', self.inertia_tensor)
	def getPos(self):
		return tuple(map(list,self.locs))

	def getTypes(self):
		return list(self.labels)

	def getTypeNums(self):
		return self.typeNums
	
	def getInertia(self):
		[principal_moments,_] = np.linalg.eig(self.inertia_tensor)
		return principal_moments
	
	def gen_rp(self, phi, R, rp):
		if phi == PatchyRings.Z_ANGLE:
			return np.array([0,0,R]).reshape((1,3))
		elif phi == 2*np.pi:
			return np.array([0,0,R])
		elif phi==np.pi:
			return np.array([0,0,-R])
		h = R*np.cos(phi)
		r = np.sqrt(R*R-h*h)
		d = 2.0*rp
		th = np.arccos((2.0*r*r-d*d)/(2.0*r*r))
		n = int(np.floor(2.0*np.pi/th))
		th = 2.0 * np.pi/n
		vec = np.zeros((n,3))
		for i in range(n):
			vec[i] = np.array([r*np.cos(th*(i+1)), r*np.sin(th*(i+1)), h])
		return vec
	
	def binary_search(self, v1, v2):
		def rmsd(v1,v2):
			ret = 0.
			for i in range(v2.shape[0]):
				norms = [np.linalg.norm(v2[i]-v1[j]) for j in range(v1.shape[0])]
				norms.sort()
				ret = ret + norms[0] + norms[1]
			ret = np.sqrt(ret/(2.0*v2.shape[0]))
			return ret

		def Rz(phi):
			return np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]])
		if (v1.shape[0] == 1):
			return v2
		err = 1e-12
		theta = 2*np.pi/v2.shape[0]
		to = 0
		th_max = theta
		tn = th_max
		go = rmsd(v1, v2)
		gn = rmsd(v1, np.dot(v2,Rz(tn)))
		while abs(tn - to) > err:
			tm = (tn+to)/2.0
			gm = rmsd(v1, np.dot(v2, Rz(tm)))
			if abs(gm - go) < abs(gm - gn):
				gn = gm
				tn = tm
			else:
				go = gm
				to = tm
		#return v2
		return np.dot(v2, Rz(tm))

