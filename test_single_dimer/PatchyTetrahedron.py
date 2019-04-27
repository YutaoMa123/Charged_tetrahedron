import numpy as np
from scipy.linalg import expm, norm


class PatchyTetrahedron():
	def __init__(self):
		print('Creating Patchy Ring Tetrahedron')

	def rotation_matrix(self,axis,phi):
		return expm(np.cross(np.eye(3), axis/norm(axis)*phi))

	def azimuth_reference(self,axis):
		v = np.array([1,0,0])
		ref = np.cross((np.cross(axis,v)),axis)
		return ref/norm(ref)

	def gen_ring(self,R,rp,cur_vertex,a,theta):
		# Generate patches at given polar angle
		axis = cur_vertex/norm(cur_vertex)
		if (np.isclose(theta,0) or np.isclose(theta,2*np.pi)):
			return (cur_vertex + axis * R).reshape((1,3))
		if (np.isclose(theta,np.pi)):
			return (cur_vertex - axis * R).reshape((1,3))
		r = R*np.sin(theta)
		d = 2*rp
		d_phi = np.arccos(1-d**2/(2*r**2))
		n = int(np.floor(2*np.pi/d_phi))
		d_phi = 2*np.pi/n
		vec = np.zeros((n,3))
		for i in range(n):
			M = self.rotation_matrix(axis,i*d_phi)
			q = np.dot(M,a)
			vec[i,:] = cur_vertex + q*r + axis*R*np.cos(theta)
		return vec



	def setup(self,types,sigma,mass,n_layers):
		# First type in types represents center of tetrahedron
		# Second type represents vertices of tetrahedron
		# Third type represents patches
		self.alpha = np.arccos(1-2*sigma[2]**2/sigma[1]**2)
		self.sigma = sigma
		self.types = types
		self.mass = mass
		self.n_layers = n_layers
		self.type_nums = [0 for i in range(len(sigma))]
		self.type_nums[0] = 1

		# Put vertices of tetrahedron in place
		self.locs = sigma[1]/np.sqrt(8./3) * np.array([[np.sqrt(8./9),0,-1./3],
			                                           [-np.sqrt(2./9),np.sqrt(2./3),-1./3],
			                                           [-np.sqrt(2./9),-np.sqrt(2./3),-1./3],
			                                           [0.,0.,1.]])
		self.labels = [types[1] for i in range(4)]
		self.type_nums[1] = 4

		# Now generate patches on vertices
		R = self.sigma[1]/2.
		rp = self.sigma[2]/2.
		for i in range(4):
			cur_vertex = self.locs[i,:]
			reference = self.azimuth_reference(cur_vertex/norm(cur_vertex))
			for j in range(n_layers):
				theta = j*self.alpha
				vec = self.gen_ring(R,rp,cur_vertex,reference,theta)
				if (j > 1):
					vec = self.binary_search(cur_vertex,align_ref,vec)
				align_ref = vec
				self.locs = np.vstack((self.locs,vec))
				self.type_nums[2] += len(vec)
				self.labels = np.hstack((self.labels,[types[2] for k in range(len(vec))]))

	def get_pos(self):
		return tuple(map(list,self.locs))

	def get_type_nums(self):
		return self.type_nums
	
	def get_labels(self):
		return list(self.labels)

	def binary_search(self,cur_vertex,v1, v2):
		def rmsd(v1,v2):
			ret = 0.
			for i in range(v2.shape[0]):
				norms = [np.linalg.norm(v2[i]-v1[j]) for j in range(v1.shape[0])]
				norms.sort()
				ret = ret + norms[0] + norms[1]
			ret = np.sqrt(ret/(2.0*v2.shape[0]))
			return ret
		axis = cur_vertex/norm(cur_vertex)
		err = 1e-12
		theta = 2*np.pi/v2.shape[0]
		to = 0
		th_max = theta
		tn = th_max
		go = rmsd(v1, v2)
		gn = rmsd(v1, np.dot(v2,self.rotation_matrix(axis,tn).T))
		print (np.isclose(go,gn))
		while abs(tn - to) > err:
			tm = (tn+to)/2.0
			gm = rmsd(v1, np.dot(v2, self.rotation_matrix(axis,tm).T))
			if abs(gm - go) <= abs(gm - gn):
				gn = gm
				tn = tm
			else:
				go = gm
				to = tm
		#return v2
		return np.dot(v2, self.rotation_matrix(axis,tm).T)










