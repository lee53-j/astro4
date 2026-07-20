import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.centroids import centroid_com
from scipy.ndimage import gaussian_filter


def find_galaxy_center(image):
    smooth = gaussian_filter(image, sigma=3)
    y, x = centroid_com(smooth)
    return x, y


def radial_surface_brightness(image, center, max_radius=None):
    x0, y0 = center

    if max_radius is None:
        max_radius = int(min(image.shape) / 2)

    radii = []
    brightness = []
    previous_flux = 0

    for r in range(2, max_radius):
        aperture = CircularAperture((x0, y0), r=r)
        phot_table = aperture_photometry(image, aperture)
        total_flux = phot_table['aperture_sum'][0]

        annulus_area = np.pi * (r**2 - (r-1)**2)
        annulus_flux = total_flux - previous_flux
        previous_flux = total_flux

        surface = annulus_flux / annulus_area

        radii.append(r)
        brightness.append(surface)

    return np.array(radii), np.array(brightness)


def cumulative_flux(image, center, max_radius=None):
    x0, y0 = center

    if max_radius is None:
        max_radius = int(min(image.shape) / 2)

    radii = []
    flux = []

    for r in range(2, max_radius):
        aperture = CircularAperture((x0, y0), r=r)
        phot_table = aperture_photometry(image, aperture)

        radii.append(r)
        flux.append(phot_table['aperture_sum'][0])

    return np.array(radii), np.array(flux)


def effective_radius(image, center):
    radii, flux = cumulative_flux(image, center)

    total = flux[-1]
    half = total / 2

    idx = np.argmin(np.abs(flux - half))

    return radii[idx], total


st.set_page_config(page_title='Galaxy Structure Analyzer', layout='wide')

st.title('🌌 Galaxy Structure Analyzer')
st.write('Analyze galaxy FITS images and calculate Surface Brightness Profile and Effective Radius.')

uploaded_file = st.file_uploader('Upload FITS Image', type=['fits', 'fit'])

if uploaded_file is not None:

    hdul = fits.open(uploaded_file)
    image = hdul[0].data.astype(float)
    hdul.close()

    if image.ndim > 2:
        image = image[0]

    st.subheader('Original FITS Image')

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(
        image,
        origin='lower',
        cmap='gray',
        vmin=np.percentile(image, 5),
        vmax=np.percentile(image, 99)
    )

    st.pyplot(fig)

    center_x, center_y = find_galaxy_center(image)

    st.success(f'Galaxy Center: ({center_x:.1f}, {center_y:.1f})')

    fig2, ax2 = plt.subplots(figsize=(6, 6))

    ax2.imshow(
        image,
        origin='lower',
        cmap='gray',
        vmin=np.percentile(image, 5),
        vmax=np.percentile(image, 99)
    )

    ax2.scatter(center_x, center_y, color='red', s=80)

    st.pyplot(fig2)

    st.subheader('Surface Brightness Profile')

    radii, brightness = radial_surface_brightness(image, (center_x, center_y))

    fig3, ax3 = plt.subplots(figsize=(7, 5))

    ax3.plot(radii, brightness, color='blue')
    ax3.set_xlabel('Radius (pixels)')
    ax3.set_ylabel('Surface Brightness')
    ax3.grid(True)

    st.pyplot(fig3)

    re, total_flux = effective_radius(image, (center_x, center_y))

    st.subheader('Results')

    col1, col2 = st.columns(2)

    with col1:
        st.metric('Effective Radius', f'{re:.1f} pixels')

    with col2:
        st.metric('Total Flux', f'{total_flux:.0f}')

    st.subheader('Effective Radius Overlay')

    fig4, ax4 = plt.subplots(figsize=(6, 6))

    ax4.imshow(
        image,
        origin='lower',
        cmap='gray',
        vmin=np.percentile(image, 5),
        vmax=np.percentile(image, 99)
    )

    circle = plt.Circle((center_x, center_y), re, color='red', fill=False, linewidth=2)

    ax4.add_patch(circle)
    ax4.scatter(center_x, center_y, color='yellow', s=60)

    st.pyplot(fig4)

st.sidebar.title('About')
st.sidebar.info(
    'Galaxy Structure Analyzer\\n\\n'
    '• FITS Upload\\n'
    '• Galaxy Center Detection\\n'
    '• Surface Brightness Profile\\n'
    '• Effective Radius Calculation'
)
