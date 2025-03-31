
import numpy as np                      # for numerical operations and array handling
import scipy.sparse as sp              # for sparse matrix construction
import scipy.sparse.linalg as spla     # for solving sparse linear systems
import matplotlib.pyplot as plt        # for plotting
import matplotlib.animation as animation  # for creating the animation
from IPython.display import HTML       # to display the animation in jupyter notebooks

# define constants using atomic units (ℏ = 1, m = 1)
hbar, m = 1.0, 1.0                     # planck constant and mass
Lx, Ly = 2.0, 1.0                     # physical dimensions of the simulation box
Nx, Ny = 500, 250                     # number of grid points in x and y
dx, dy = Lx / Nx, Ly / Ny            # spatial resolution
dt = 0.0005                          # time step for simulation

# define the spatial grid
x = np.linspace(-Lx/2, Lx/2, Nx)     # 1d grid in x direction
y = np.linspace(-Ly/2, Ly/2, Ny)     # 1d grid in y direction
X, Y = np.meshgrid(x, y)            # create 2d meshgrid for x and y

# initialize the potential energy matrix with a double-slit barrier
V = np.zeros((Nx, Ny))               # initialize potential as zero everywhere
barrier_x = int(0.5 * Nx)            # x-location of the barrier
V[barrier_x:barrier_x + 5, :] = 100.0  # add a vertical wall as the main barrier

# carve two slits in the barrier by zeroing regions in it
slit_width, slit_separation = int(0.05 * Ny), int(0.2 * Ny)  # define slit size and separation
V[barrier_x:barrier_x + 5, Ny//2 - slit_separation:Ny//2 - slit_separation + slit_width] = 0
V[barrier_x:barrier_x + 5, Ny//2 + slit_separation - slit_width:Ny//2 + slit_separation] = 0

# define initial wave packet (a gaussian modulated by a complex phase)
x0, y0 = -0.7, 0                     # initial position of the wave packet
k0x, k0y = 20.0, 0.0                 # initial momentum (directional)
sigma = 0.1                          # width of the gaussian
A = (1 / (2 * np.pi * sigma**2))**0.5   # normalization constant
psi = A * np.exp(-((X - x0)**2 + (Y - y0)**2) / (4 * sigma**2)) * np.exp(1j * (k0x * X + k0y * Y))
psi_flat = psi.ravel()               # flatten wavefunction into 1d array for matrix operations

# precompute decoherence phase noise (applied multiplicatively each step)
decoherence_noise = np.exp(1j * 0.0001 * np.random.randn(Nx * Ny))

# function to apply decoherence at each step
def apply_decoherence(psi_flat):
    return psi_flat * decoherence_noise

# construct the sparse laplacian operator for 2d finite-difference scheme
def laplacian(Nx, Ny, dx, dy):
    Dxx = sp.diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)) / dx**2  # second derivative in x
    Dyy = sp.diags([1, -2, 1], [-1, 0, 1], shape=(Ny, Ny)) / dy**2  # second derivative in y
    return sp.kron(sp.eye(Ny), Dxx) + sp.kron(Dyy, sp.eye(Nx))      # build full 2d laplacian

# build the hamiltonian operator using the kinetic and potential terms
H = (-hbar**2 / (2 * m)) * laplacian(Nx, Ny, dx, dy) + sp.diags(V.ravel(), 0)

# prepare matrices for the crank-nicolson time evolution scheme
I = sp.eye(Nx * Ny, format='csr')                          # identity matrix
A = (I + 1j * dt * H / (2 * hbar)).tocsr()                 # left-hand matrix in crank-nicolson
B = (I - 1j * dt * H / (2 * hbar)).tocsr()                 # right-hand matrix
A_inv = spla.factorized(A)                                # pre-factorize A for efficient reuse

# define one time step of the evolution, including decoherence
def step(psi_flat):
    return apply_decoherence(A_inv(B @ psi_flat))         # advance one step with decoherence

# setup plot style and figure for visualization
plt.style.use('dark_background')
fig, axes = plt.subplots(1, 3, figsize=(18, 6))            # create side-by-side subplots
plt.subplots_adjust(wspace=0.05)                          # reduce space between plots

# choose color maps
cmap_real_imag = "twilight_shifted"
cmap_prob = "inferno"

# initialize images for real, imaginary, and probability density views
psi_reshaped = psi_flat.reshape((Nx, Ny))
im_real = axes[0].imshow(np.real(psi_reshaped), cmap=cmap_real_imag, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
axes[0].set_title("Real Part of $\Psi(x,y,t)$")

im_imag = axes[1].imshow(np.imag(psi_reshaped), cmap=cmap_real_imag, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
axes[1].set_title("Imaginary Part of $\Psi(x,y,t)$")

im_prob = axes[2].imshow(np.abs(psi_reshaped)**2, cmap=cmap_prob, extent=[-Lx/2, Lx/2, -Ly/2, Ly/2])
axes[2].set_title("Probability Density $|\Psi(x,y,t)|^2$")

# label axes
for ax in axes:
    ax.set_xlabel("$x$ [a.u.]")
    ax.set_ylabel("$y$ [a.u.]")

# update function for animation; evolves the system and updates plots
def update(frame):
    global psi_flat
    for _ in range(5):                           # perform several small steps per frame for smoother animation
        psi_flat = step(psi_flat)
    psi_reshaped = psi_flat.reshape((Nx, Ny))    # reshape back to 2d for plotting
    im_real.set_data(np.real(psi_reshaped))      # update real part
    im_imag.set_data(np.imag(psi_reshaped))      # update imaginary part
    im_prob.set_data(np.abs(psi_reshaped)**2)    # update probability density
    return [im_real, im_imag, im_prob]

# create and save the animation
ani = animation.FuncAnimation(fig, update, frames=300, interval=15, blit=False)
ani.save("GMWdouble_slit_experiment4.gif", writer="pillow", fps=15, dpi=150)

# display animation inline in jupyter
display(HTML(ani.to_jshtml()))
