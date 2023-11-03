import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

rc = {'figure.figsize':(8,4.5),
        'axes.facecolor':'#0e1117',
        'axes.edgecolor': '#0e1117',
        'axes.labelcolor': 'white',
        'figure.facecolor': '#0e1117',
        'patch.edgecolor': '#0e1117',
        'text.color': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'grid.color': 'grey',
        'font.size' : 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12}
plt.rcParams.update(rc)

st.set_page_config(page_title="Gaussian Beam Profile", initial_sidebar_state="collapsed")



st.title("Gaussian Beam Profile")
st.markdown("Code provided by Tapio Simula")

col1, col2 = st.columns([1,2])

with col1:
    # Define the beam parameters
    w0 = st.slider("waist size (mm)", 0.1, 10.0, 2.0)     # waist size
    zR = st.slider("Rayleigh range", 0.1, 20.0, 10.0)    # Rayleigh range
    z  = st.slider("Position along z axis (cm)", 0.1, 100.0, 0.0)     # position along beam axis
    wavelength = st.number_input("Wavelength of incident beam (mm)", 100, 1000, 700, step=50) # wavelength
    k  = 2 * np.pi / (wavelength * 1e-9)   # wavenumber


# Define the coordinate system
xmin, xmax, ymin, ymax = -10, 10, -10, 10
x    = np.linspace(xmin, xmax, 200)
y    = np.linspace(ymin, ymax, 200)
X, Y = np.meshgrid(x, y)


# Compute the intensity profile
R2  = (X**2 + Y**2)
w   = w0 * np.sqrt(1 + (z/zR)**2)
R   = np.sqrt(R2 + (w**2)*(z/zR)**2)
psi = np.arctan2(np.sqrt(R2), (z*zR))
E   = np.exp(-R2/w**2) * np.exp(-1j * k * z) * np.exp(1j * k * R**2 / (2*zR))
I   = np.abs(E * np.conj(E))
I   = I / np.max(I)

slice_idx = 100
I1D       = I[:, slice_idx]

# Plot the results
#fig   = plt.figure()
fig, ax   = plt.subplots(2,1,sharex=True,figsize=(6,6), gridspec_kw={'height_ratios': [2, 1]})
cbar = ax[0].imshow(I, cmap='hot',extent=[xmin, xmax, ymin, ymax])
ax[0].set_xlabel('x (mm)')
ax[0].set_ylabel('y (mm)')
plt.colorbar(cbar)


ax[1].plot(x, I1D)
ax[1].set_xlabel('x (mm)')
ax[1].set_ylabel('I ($I_0$)')

y_e2   = 1 / np.e**2
y_fwhm = 1 / 2
idx    = np.argmin(np.abs(I1D - y_e2))
x_e2   = x[idx]
ax[1].axvline(x_e2, linestyle='--', color='gray')
ax[1].axvline(-x_e2, linestyle='--', color='gray')
ax[1].axhline(y_e2, linestyle='--', color='gray')

with col2:
    st.pyplot(fig)