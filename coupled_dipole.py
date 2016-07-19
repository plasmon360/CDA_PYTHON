from LD import LD  # Make sure the file is accessible to PYTHONPATH or in the same directory of file which is trying to import
# try to import it from https://github.com/plasmon360/LD_python
import time
import numpy as np
from numpy import pi, exp, dot, conj, sum, mean, imag
from numpy.linalg import inv, norm
from numpy.random import random_integers, uniform, randn
from scipy.sparse.linalg import isolve
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)


def create_particle_on_grid(nx=1, ny=1, step_x=100E-9, step_y=100E-9,radius=15E-9, plot = True):
    """
    This function creates particles on a rectagular grid, nx and ny control the number along the x and y axis,
    step_x and step_y is periodicity in x and y (nm), radius is the radius of the sphere. If you want to visualize these spheres in mayavi, plot is enable true

    """
    assert (nx >= 1 and ny >= 1), 'Make sure nx and ny is >= 1'
    N = nx * ny # total number of sphere

    # Cretes the numpy vector of position coordinate. each position coordinate is vector of pos_x, pos_y, pos_z
    pos = np.zeros([N, 3]) # A vector of vectors , (x,y,z) positions
    # Creates a effective radius vector
    r_eff = np.zeros([N]) # A vector with effective radius of each particle

    # Assign the posititio coordinates and radius to each particle
    kk = 0
    for pos_i in np.arange(nx):
        for pos_j in np.arange(ny):
            pos[kk] = np.array([pos_i * step_x, pos_j * step_y, 0])
            r_eff[kk] = radius
            kk += 1

    # Plot in Mayavi
    if plot:
        from mayavi import mlab
        fig = mlab.figure(size=(1400,600))
        pts = mlab.points3d(pos[:,0],pos[:,1],pos[:,2],r_eff*2,scale_factor = 1,resolution=15)
        mlab.show()
    return N, pos, r_eff

def random_particles_not_touching(N=10, x_span=100E-9, y_span=100E-9,z_span=40E-9, max_radius = 10E-9, min_radius = 2E-9,plot=False):
    """
    Here we create a particles with random positions and radius that are not touching in a box
    :param N: number of particles
    :param x_span: x span of the bounding box (nm)
    :param y_span: y span of the bounding box (nm)
    :param z_span: z span of the bounding box(nm)
    :param max_radius: the maximum radius for the a randome  sphere (nm)
    :param min_radius: the minimum radius for a random sphere(nm)
    :param plot: if we want to plot in mayavi for visualization
    :return: Returns the number of particles, vector containing vector of position coordinates of spheres, vector containing the effective radii of spheres
    """
    assert (4/3)*np.pi*max_radius**3 <= x_span*y_span*z_span # Make sure the volume of the box is larger than the volume of the largest sphere
    pos = np.zeros([N, 3]) # A vector of vector (x,y,z) positions
    r_eff = np.zeros([N]) # A vector with effective radius of each particle

    # calcualtes the distance between two spheres
    def distance (pos_1,pos_2):
        temp = pos_2-pos_1
        return np.sqrt(temp[0]**2 + temp[1]**2 +temp[2]**2)

    i = 0
    while (i < N):
        # Create a position and radius that is uniformly distributed
        pos[i] = uniform(low =0, high = 1, size = 3)* np.array([x_span, y_span, z_span])
        r_eff[i] = uniform(low =min_radius/max_radius, high = 1, size = 1)* max_radius
        # Lets check if the distance between the current sphere and all other spheres is less than the sum of radius of the current sphere and all other spheres
        for j in xrange(i):
             if (distance(pos[j],pos[i]) < 1*(r_eff [i]+r_eff [j])) :
                print distance(pos[j],pos[i]), 1*(r_eff [i]+r_eff [j])
                print 'Collision with other sphere detected. Removing this sphere'
                i=i-1 # Lets remove this guy and start our again
                break

        i+=1

    # plot the spheres in mayavi
    if plot:
        from mayavi import mlab
        fig = mlab.figure(size=(1400,600))
        pts = mlab.points3d(pos[:,0],pos[:,1],pos[:,2],r_eff*2, scale_factor = 1)
        pts.glyph.glyph.clamping = False # THis is needed because the smallest sphere would be so small that you can see, http://docs.enthought.com/mayavi/mayavi/mlab.html
        mlab.show()

    return N, pos, r_eff



# # we will plot the data here
# from mayavi import mlab
# fig = mlab.figure(size=(1400,600))
# pts = mlab.points3d(pos[:,0],pos[:,1],pos[:,2],rad*2, scale_factor = 1)
# pts.glyph.glyph.clamping = False # THis is needed because the smallest sphere would be so small that you can see, http://docs.enthought.com/mayavi/mayavi/mlab.html
# mlab.show()

def calc_alpha_i(r, eps, eps_surround):
    """
    :param r_eff: effective radius of the i th particle
    :param eps: Dielectric function at a certain wavelength
    :param eps_surround: Dielectric of surrounding at certain wavelength
    :return: calculate the polarlization tensor for sphere
    """
    ax = (3 * (2 * r) ** 3) / (4 * pi) * (eps - eps_surround) / (eps + 2 * eps_surround)
    az = ay = ax
    alpha_i = (np.array([[ax, 0, 0], [0, ay, 0], [0, 0, az]]))
    return alpha_i


def Aij_matrix(arguments):
    """
    Calculates the interaction matrix between two particles
    :param arguments[0]: Aij
    :param arguments[1]: wave vector
    :param arguments[2]: vector with x,y,z location of particle i
    :param arguments[3]: vector with x,y,z location of particle j
    :return: Aij, 3x3 interaction matrix between two particles
    """

    Aij = arguments[0]
    k = arguments[1]
    pos_i=arguments[2]
    pos_j =arguments[3]

    temp = pos_j - pos_i
    r_ij = np.sqrt(temp[0]**2+temp[1]**2+temp[2]**2)
    # calculate the unit vectors between two particles
    nx, ny, nz = temp / r_ij
    eikr = exp(1j * k * r_ij)

    A = (k ** 2) * eikr / r_ij
    B = (1 / r_ij ** 3 - 1j * k / r_ij ** 2) * eikr
    C = 3*B-A

    Aij[0][0] = A * (ny ** 2 + nz ** 2) + B * (3 * nx ** 2 - 1)
    Aij[1][1] = A * (nx ** 2 + nz ** 2) + B * (3 * ny ** 2 - 1)
    Aij[2][2] = A * (ny ** 2 + nx ** 2) + B * (3 * nz ** 2 - 1)

    Aij[0][1] = Aij[1][0] = nx * ny * C
    Aij[0][2] = Aij[2][0] = nx * nz * C
    Aij[1][2] = Aij[2][1] = ny * nz * C

    return Aij


def calc_A_matrix(A, k, N, r_eff, epsilon, eps_surround):
    """
    This function calcualtes the A matrix for N particles
    :param A: 3N x 3N matrix filled with zeros - input
    :param k: wavevector
    :param N: number of spheres
    :param r_eff: vector containing the effective radius of the spheres
    :param epsilon: dielectric function of the metal
    :param eps_surround: dielectric function of the surrounding
    :return: Returns A filled with interaction terms between particles
    """

    Aij = np.zeros([3, 3], dtype=complex) # interaction matrix

    for i in xrange(N):
        for j in xrange(N):
            if i == j:
                # diagonal of the matrix
                A[3 * i: 3 * (i + 1), 3 * j: 3 * (j + 1)] = inv(calc_alpha_i(r_eff[i], epsilon, eps_surround))
            else:

                A[3 * i: 3 * (i + 1), 3 * j: 3 * (j + 1)] = -1 * Aij_matrix([Aij, k, pos[i], pos[j]])



def calc_E_inc(k_vec_k, N, pos, E0, E_inc):
    """
    Calculates the incident electric field at the centers of each particles
    :param k_vec_k: k_unit vector * k
    :param N: number of the particles
    :param pos: vector containing the position of the particles
    :param E0: Incident electric field vector
    :param E_inc: prepopulate vector of 3N size filled with zeros
    :return: E_inc: vector of 3N filled with incident electric field
    """
    for i in xrange(N):
        E_inc[3 * i: 3 * (i + 1)] = E0[3 * i: 3 * (i + 1)] * exp(1j * dot(k_vec_k, pos[i]))



if __name__ == '__main__':
    # here is the main calculation
    # Lets define some parameters for the simulation
    start_wave = 320.0E-9
    end_wave = 500.0E-9
    wave_spacing = 10.0E-9
    solver_tolerance = 1E-10

    # Lets prepare the wavelength vector and dielectric function
    wave = np.arange(start_wave, end_wave, wave_spacing)
    print wave

    eps = LD(wave, material='Ag', model='LD')  # Creates gold object with dielectric function of LD model
    n_surround = 1 # refractive index of the surrounding

    E0_vec = np.array([1, 0, 0], dtype=complex)  # incident polarization of the light, [1,0,0] means light is polarized in x, [0,1,0] mean light is polarized in y
    k_vec = np.array([0, 0, 1]) # inciddent k vector [0,0,1] means light is travelling in +z axis

    eps_surround = n_surround ** 2 # Dielectric function of surro

    # Create particles on grid
    N, pos, r_eff = create_particle_on_grid(nx=10, ny=10, step_x=2*30E-9, step_y=1.2*30E-9,radius=15E-9, plot = True)


    # # Create particles of random diameter and position
    # N, pos, r_eff = random_particles_not_touching(N=200, x_span=100E-9, y_span=100E-9,z_span=40E-9, max_radius = 10E-9, min_radius = 2E-9,plot=True)


    # pre allocate the arrays/vectors

    A = np.zeros([3 * N, 3 * N], dtype='complex') # A matrix of 3N x 3N filled with zeros
    p = np.zeros(3 * N, dtype='complex')  # A vector that stores polarization Px, Py, Pz of the each particles, we will use this for initial guess for solver
    E0 = np.tile(E0_vec, N) # A vector that has the Ex, Ey, Ez for each particles, we make this by tiling the E0_vec vector
    E_inc = np.zeros(3 * N, dtype='complex') # A vector storing the Incident Electric field , Einc_x, Einc_y, Einc_z for each particle


    n_wave = wave.size
    p_calc = np.zeros([3 * N, n_wave], dtype='complex')  # This stores the dipoles moments for each particle at different wavelengths
    c_ext = np.zeros(n_wave) # stores the extinction crosssection
    c_abs = np.zeros(n_wave) # stores the absorption crossection
    c_scat = c_ext-c_abs # stores the scattering crossection


    start = time.clock()

    # loop over the wavelength

    for w_index, w in enumerate(wave):
        start_at_this_wavelength = time.clock()

        print '-'*100
        print 'Running wavelength: ', w * 1E9

        k = (2 * pi / w)  # create the k

        epsilon = eps.epsilon[w_index] # Get the dielectric function

        # calc_A_matrix_parallel(A, k, N, r_eff, epsilon, eps_surround) # Calculate the A matrix
        calc_A_matrix(A, k, N, r_eff, epsilon, eps_surround) # Calculate the A matrix

        calc_E_inc(k_vec*k, N, pos, E0, E_inc)

        # We will define a callback that calculates the current residual
        iter = 1

        def mycallback(xk):
            # here xk is current solution of the iteration
            global iter
            # residual is defined as norm( E_inc - A*xk ) / norm( E_inc)
            residual = norm(E_inc - dot(A, xk)) / norm(E_inc)
            print("%s : %0.3E" % (iter, residual))
            iter += 1


        # Solve AP = E, where A = N x N complex matrix, P and E are N vectorusing biconjugate gradient method

        print("Iteration : Residual")
        p_calc[:,w_index], info = isolve.bicgstab(A, E_inc, callback=mycallback, x0=p, tol=solver_tolerance, maxiter=None)

        if info == 0:
            print 'Successful Exit'
            # calculate the extinction crossection
            c_ext[w_index] = (4 * pi * k / norm(E0) ** 2) * np.sum(np.imag(dot(conj(E_inc), p_calc[:, w_index])))

            # calculate the absorption crossection
            for i in xrange(N):
                c_abs[w_index] += ( np.imag(dot(p_calc[3*i : 3*(i+1), w_index],
                                                    dot(conj(A[3 * i: 3 * (i + 1), 3 * i: 3 * (i + 1)]),
                                                        conj(p_calc[3*i: 3*(i+1), w_index]))
                                               )
                                          )
                                  - (2.0/3)*k**3*norm(p_calc[3*i : 3*(i+1), w_index])**2
                                  )
            c_abs[w_index] *= (4 * pi * k / norm(E0) ** 2)
            c_scat[w_index] = c_ext[w_index] - c_abs[w_index]

        elif info > 0:
            print 'Convergence not achieved, may be increase the number of maxiter'

        elif info < 0:
            print info
            print 'illegal input'

        end_at_this_wavelength = time.clock()

        print("Elapsed Time @ this wavelenth %3.2f sec" % (end_at_this_wavelength - start_at_this_wavelength))

    # Calculate the extinction efficiency. The effective area is the mean of individual particle areas
    def efficiency_calc(cross_section, r_eff):
        return cross_section / (mean(pi * r_eff ** 2))

    q_ext = efficiency_calc(c_ext, r_eff)
    q_abs = efficiency_calc(c_abs, r_eff)
    q_scat = efficiency_calc(c_scat, r_eff)

    end = time.clock()
    print("Elapsed Time %3.2f sec" % (end - start))

    # we will plot the data here
    f, ax = plt.subplots(1,1)
    ax.plot(wave * 1E9, q_ext, '-r', label = 'Extinction')
    ax.plot(wave * 1E9, q_abs, '-b', label = 'Absorption')
    ax.plot(wave * 1E9, q_scat,'-g', label = 'Scattering')
    ax.legend(loc = 'upper right')
    ax.set_ylabel('Efficiency')
    ax.set_xlabel('wavelength (nm)')
    plt.grid('on')
    plt.show()

    # we will save the data here
    import h5py
    import os
    dir =  os.path.dirname(os.path.realpath(__file__))
    f = h5py.File(os.path.join(dir,"simulation_data.hdf5"), "w")
    f.create_dataset('wave', data=q_scat)
    f.create_dataset('q_abs', data=q_scat)
    f.create_dataset('q_ext', data=q_scat)
    f.create_dataset('q_scat', data=q_scat)
    f.create_dataset('p_calc', data = p_calc)
    f.create_dataset('pos', data = pos)
    f.create_dataset('r_eff', data = r_eff)
    f.close()

    ## If we want to plot the dipole moments in mayavi
    # from mayavi import mlab
    # fig = mlab.figure(size=(1400,600))
    # pts1 = mlab.points3d(pos[:,0],pos[:,1],pos[:,2],r_eff*2,scale_factor = 1,resolution=15, opacity = 0.5)
    # pts2 = mlab.quiver3d( pos[:,0],pos[:,1],pos[:,2], np.abs(p_calc[0::3, 1]), np.abs(p_calc[1::3, 1]), np.abs(p_calc[2::3, 1]) )
    # pts1.glyph.glyph.clamping = False # THis is needed because the smallest sphere would be so small that you can see, http://docs.enthought.com/mayavi/mayavi/mlab.html
    # mlab.show()