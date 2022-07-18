from __future__ import division
from dolfin import *
import mshr
import numpy as np
from petsc4py import PETSc
import sys,argparse
import os

# This code implements the numerical method described in
#
# E. S. Gawlik & F. Gay-Balmaz. A Conservative Finite Element Method for the 
# Incompressible Euler Equations with Variable Density. Journal of 
# Computational Physics, 412, 109439 (2020).
#
# Example usage:
# python3 variabledensity.py -o 0 -d 1 -t 5.0 -k 0.01 -a 0.5 -b 0.5 -r 3


# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;
#parameters["num_threads"] = 2

if __name__ == "__main__":

	total = len(sys.argv)
	cmdargs = str(sys.argv)
	print ("The total numbers of args passed to the script: %d " % total)
	print ("Args list: %s " % cmdargs)
	print ("Script name: %s" % str(sys.argv[0]))

	parser = argparse.ArgumentParser()
	parser.add_argument('-r', '--num_refinements', default=0, type=int)
	parser.add_argument('-o', '--order', default=0, type=int)
	parser.add_argument('-d', '--order_density', default=0, type=int)
	parser.add_argument('-k', '--dt', default=0.1, type=float)
	parser.add_argument('-t', '--T', default=1.0, type=float)
	parser.add_argument('-a', '--alpha', default=0.1, type=float)
	parser.add_argument('-b', '--beta', default=0.1, type=float)
	parser.add_argument('-s', '--savesoln', default=False, type=bool)
	args = parser.parse_args()
	num_refinements = args.num_refinements
	order = args.order
	order_density = args.order_density
	dt = args.dt
	T = args.T
	alpha = args.alpha
	beta = args.beta
	savesoln = args.savesoln

	directory = 'results/variabledensity'
	directory = directory + '_o' + str(order) + 'd' + str(order_density) + 'r' + str(num_refinements) + '/'
	if not os.path.exists(directory):
		os.makedirs(directory)
	class Logger(object):
		def __init__(self):
			self.terminal = sys.stdout
			self.log = open(directory + "output.log", "w+")
			
		def write(self, message):
			self.terminal.write(message)
			self.log.write(message)  
				
		def flush(self):
			pass    
	
	sys.stdout = Logger()


	print("order = %i" % order)
	print("order_density = %i" % order_density)
	print("T = %f" % T)
	print("dt = %f" % dt)
	print("alpha = %f" % alpha)
	print("beta = %f" % beta)
	print("num_refinements = %i" % num_refinements)
	if savesoln:
		print("Solution will be saved.")


	# Define problem parameters 
	
	# Space dimension
	space_dim = 2

	# Quadrature degree
	quad_degree = 6#max(order,order_density) + 2
	parameters["form_compiler"]["quadrature_degree"] = quad_degree
	
	# Get communicator and process rank
	comm = 0;#mpi_comm_world()
	#mpiRank = MPI.rank(comm)
	
	#########################################
	# Set problem parameter values

	rhomin = 1.0
	rhomax = 3.0
	d = 1.0
	gravity = 10.0

	# Domain size
	length = d 
	height = 4.0*d

	###################################################
	# Construct the mesh
	nx = 2**(num_refinements+1)
	ny = 4 * 2**(num_refinements+1)
	mesh = RectangleMesh(Point(-length/2, -height/2), Point(length/2, height/2), nx, ny, 'crossed')
	
	# Compute domain area and average h
	area = assemble(1.0*dx(mesh, degree=2))
	havg = (area / mesh.num_cells())**(1.0 / mesh.topology().dim())

	print("area = ",area)
	print("havg = ",havg)
	print("hmax = ",mesh.hmax())
	print("hmin = ",mesh.hmin())


	#for i in range(num_refinements):
	if 0:
		print("refining, step number ",i)
		# Mark cells for refinement
		markers = MeshFunction("bool", mesh, mesh.topology().dim())
		markers.set_all(False)
		for cell in cells(mesh):
			if True:#cell.midpoint().distance(center) > 0.9:
				markers[cell.index()] = True

		# Refine mesh
		mesh = refine(mesh, markers)

	

	########################################
	# Setup output files
	xdmf = 0
	
	ext = '.xdmf' if xdmf else '.pvd'
	files_names = [ 'fluidVI_%id_u','fluidVI_%id_D','fluidVI_%id_p']
	files_names = map(lambda x: directory + x%space_dim + ext , files_names )
	
	files = [] 
	for file_name in files_names:
	
		if xdmf :
			if MPI.size(comm) == 1:
				files.append(XDMFFile(file_name))
			else:
				files.append( XDMFFile(mesh.mpi_comm(),file_name) )
			files[-1].parameters['flush_output'] = True
			files[-1].parameters['rewrite_function_mesh'] = False
		else :
			files.append( File(file_name, 'compressed') )

	
	###############################
	# Fe space
	# Note: This code assumes 2*order <= order_density, so that no L2 projection of u \cdot u is needed.
	# It also assumes order_density >= 1, so that no L2 projection of the gravitational potential is needed.
	Ve = FiniteElement("RT", mesh.ufl_cell(), order+1)
	Fe = FiniteElement("DG", mesh.ufl_cell(), order_density)
	Qe = FiniteElement("DG", mesh.ufl_cell(), order)
	VFQ = FunctionSpace(mesh, MixedElement((Ve, Fe, Qe)))

	n = FacetNormal(mesh)

	uDpf = Function(VFQ)
	(v,E,q) = TestFunctions(VFQ)
	(du,dD,dp) = TrialFunctions(VFQ)
	vEqg = (v,E,q)
	dudDdpdf = (du,dD,dp)
	(u,D,p) = split(uDpf)

	uDp0 = Function(VFQ)
	uDp1 = Function(VFQ)
	(u0,D0,p0) = split(uDp0)
	(u1,D1,p1) = split(uDp1)

	# Create files for storing solution
	uDp1.rename('uDpf','uDpf')


	w = Function(VFQ)   
	nullspace_basis = w.vector().copy()
	VFQ.sub(2).dofmap().set(nullspace_basis, 1.0)
	nullspace_basis.apply('insert')
	nullspace = VectorSpaceBasis([nullspace_basis])
	nullspace.orthonormalize()

	
	print("ndof = ",uDpf.vector().size())


	# Useful functions
	def myabs(x):
		return abs(x)
		#return 2.0/pi * x * atan(x/1e-2)

	def mysign(x):
		return x/abs(x)
		#return 2.0/pi * atan(x/1e-2)


	################################################
	# Boundary and initial conditions

	# Define function G such that G \cdot n = g
	class BoundarySource(UserExpression):
		def __init__(self, mesh, **kwargs):
			self.mesh = mesh
			super().__init__(**kwargs)
		def eval_cell(self, values, x, ufc_cell):
			cell = Cell(self.mesh, ufc_cell.index)
			n = cell.normal(ufc_cell.local_facet)
			g = 0.0#*sin(pi*x[0]*2./length)*(1-exp(-self.t))
			values[0] = g*n[0]
			values[1] = g*n[1]
		def value_shape(self):
			return (2,)

	bd_u_val = BoundarySource(mesh)

	def OnBoundaryQ(x, on_boundary):
		return on_boundary

	bd_boundary_vel =  DirichletBC(VFQ.sub(0), bd_u_val, OnBoundaryQ)
	bcu = [bd_boundary_vel]

	rhoterm = 1.0
	e_u0 = Constant(('0.0','0.0'))
	e_D0 = Expression('2.0+tanh( (x[1]+0.1*d*cos(2.0*pi*x[0]/d)) / (0.1*d) )', degree=6, d=d)
	e_p0 = Constant('0.0')

	u00 = interpolate(e_u0, VFQ.sub(0).collapse())
	D00 = interpolate(e_D0, VFQ.sub(1).collapse())
	p00 = interpolate(e_p0, VFQ.sub(2).collapse())

	# Impose boundary conditions on initial condition
	assign(uDp0, [u00, D00, p00])
	uDp1.assign(uDp0)


	##########################################
	# Define the problem operators 

	# Time-stepping
	#dt = 1.e-1
	t = 0.0
	step = 0	
	max_steps = 500
	#T = dt*max_steps

	# Define coefficients
	k = Constant(dt)  

	dt_record = dt
	interval_record =  dt_record/dt
	
	force = Expression(('0.0','-gravity'),gravity=gravity,degree=0)
	ycoord = Expression('x[1]',degree=1)

	################################
	# Start time integration
	while t < T + DOLFIN_EPS:

		################################################
		# Save to file
		if not step%interval_record :
			files[0] << ( uDp1.sub(0), t )
			files[1] << ( uDp1.sub(1), t )
			files[2] << ( uDp1.sub(2), t )

		
		
		mass = assemble(D0*dx)
		rho2 = assemble(D0*D0*dx)
		energy = assemble(0.5*inner(D0*u0, u0)*dx + D0*gravity*ycoord*dx)
		divnorm = assemble(div(u0)*div(u0)*dx)
		if step <= 0:
			mass0 = mass
			rho20 = rho2
			energy0 = energy
			print("massratio = ")
			print(1.0)
			print("rho2ratio = ")
			print(1.0)
			print("energyratio = ")
			print(1.0)
		else:
			print("massratio = ")
			print(mass/mass0)
			print("rho2ratio = ")
			print(rho2/rho20)
			print("energyratio = ")
			print(energy/energy0)

		print("divergence = ")
		print(divnorm)

		bd_u_val.t = t+dt

		k.assign( dt )

		if step > max_steps:
			break


		################################################
		# Compute new velocity, density, and pressure

		# Set up equation to solve
		nn = n('+')
		u = 0.5*(u0+u1)
		D = 0.5*(D0+D1)
		Du = 0.5*(D0*u0+D1*u1)
		w = avg(Du)
		f1 = dot(u0,u1)

		F1 = (1./k)*inner(D1*u1 - D0*u0, v)*dx \
		    + inner( Du, -grad(v)*u + grad(u)*v )*dx \
		    + (nn[0]*w[1]-nn[1]*w[0]) * jump(u[0]*v[1]-u[1]*v[0])  * dS \
		    - 0.5*dot(v,grad(f1)) * D * dx \
		    + 0.5*dot(v('+'),n('+')) * jump(f1) * avg(D) * dS \
		    - p1*div(v)*dx - div(u1)*q*dx \
		    + (1./k)*(D1-D0)*E*dx - dot(u,grad(E))*D*dx \
		    + dot(u('+'),n('+')) * jump(E) * avg(D) * dS \
		    - dot(D*force,v)*dx
		
		if alpha>0.0 and t>0.0:
			ww = jump(D*u)
			F1 = F1 + alpha * mysign(dot(u('+'),n('+'))) * (nn[0]*ww[1]-nn[1]*ww[0]) * jump(u[0]*v[1]-u[1]*v[0])  * dS

		if beta>0.0 and t>0.0:
			F1 = F1 + beta * myabs(dot(u('+'),n('+'))) * jump(E)*jump(D)*dS
			F1 = F1 + 0.5*beta * mysign(dot(u('+'),n('+'))) * dot(v('+'),n('+')) *(jump(f1)*jump(D)) * dS

		# Compute Jacobian of F1
		J = derivative(F1, uDp1)
		

		bcs_du = bcu#homogenize(bcu)
		U_inc = Function(VFQ)
		nIter = 0
		eps = 1

		ksp = PETScKrylovSolver("bicgstab","ilu")
		ksp.parameters["absolute_tolerance"] = 1e-14
		ksp.parameters["relative_tolerance"] = 1e-14
		ksp.parameters["maximum_iterations"] = 400
		ksp.parameters["monitor_convergence"] = False
		ksp.parameters["error_on_nonconvergence"] = False

		while eps > 1e-10 and nIter < 10:              # Newton iterations
			nIter += 1
			A, b = assemble_system(J, -F1, bcs_du)
			ksp.set_operator(A)
			ksp.set_from_options()
			as_backend_type(A).set_nullspace(nullspace)
			nullspace.orthogonalize(b);
			if b.norm('l2') < 1e-11:
				break
			ksp.solve(U_inc.vector(), b)
			eps = np.linalg.norm(U_inc.vector(), ord = 2)
			fnorm = b.norm('l2')
			uDp1.vector()[:] += U_inc.vector()
			print("      {0:2d}  {1:3.2E}  {2:5e}".format(nIter, eps, fnorm))


		if savesoln and t+dt>T:
			# Save solution
			output_file = HDF5File(mesh.mpi_comm(), directory + "u.h5", "w")
			output_file.write(uDp0.sub(0), "solution")
			output_file.close()
			
			output_file = HDF5File(mesh.mpi_comm(), directory + "D.h5", "w")
			output_file.write(uDp0.sub(1), "solution")
			output_file.close()
			
			output_file = HDF5File(mesh.mpi_comm(), directory + "p.h5", "w")
			output_file.write(uDp0.sub(2), "solution")
			output_file.close()
			
			output_file = HDF5File(mesh.mpi_comm(), directory + "uDp.h5", "w")
			output_file.write(uDp0, "solution")
			output_file.close()
			

		# Move to next time step
		uDp0.assign(uDp1) 
		t += dt
		if 1:
			print("t =", t)
			step += 1
			

