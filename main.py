from profile import save_results
from utils import normalize_image
import numpy as np
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.centroids import centroid_com
from scipy.ndimage import gaussian_filter


def find_galaxy_center(image):
    """
    은하 중심 자동 검출
    """
    smooth = gaussian_filter(image, sigma=3)

    y, x = centroid_com(smooth)

    return x, y


def radial_surface_brightness(image, center, max_radius=None):
    """
    Surface Brightness Profile 계산
    """

    y0, x0 = center

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
    """
    반지름별 누적 Flux 계산
    """

    y0, x0 = center

    if max_radius is None:
        max_radius = int(min(image.shape)/2)

    radii = []
    flux = []

    for r in range(2, max_radius):

        aperture = CircularAperture((x0, y0), r=r)

        phot_table = aperture_photometry(image, aperture)

        radii.append(r)

        flux.append(phot_table['aperture_sum'][0])

    return np.array(radii), np.array(flux)


def effective_radius(image, center):
    """
    Half-light Radius 계산
    """

    radii, flux = cumulative_flux(image, center)

    total = flux[-1]

    half = total / 2

    idx = np.argmin(np.abs(flux-half))

    return radii[idx], total
    import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

from photometry import (
    find_galaxy_center,
    radial_surface_brightness,
    effective_radius,
)

st.set_page_config(
    page_title="Galaxy Structure Analyzer",
    layout="wide"
)

st.title("🌌 Galaxy Structure Analyzer")
st.write(
    "Analyze Surface Brightness Profile and Effective Radius from FITS images."
)

uploaded_file = st.file_uploader(
    "Upload FITS Image",
    type=["fits", "fit"]
)

if uploaded_file is not None:

    hdul = fits.open(uploaded_file)

    image = hdul[0].data.astype(float)

    hdul.close()

    if image.ndim > 2:
        image = image[0]

    st.subheader("Original FITS Image")

    fig, ax = plt.subplots(figsize=(6,6))

    ax.imshow(
        image,
        origin="lower",
        cmap="gray",
        vmin=np.percentile(image,5),
        vmax=np.percentile(image,99)
    )

    st.pyplot(fig)

    st.write("---")

    st.subheader("Finding Galaxy Center...")

    center_x, center_y = find_galaxy_center(image)

    st.success(
        f"Galaxy Center : ({center_x:.2f}, {center_y:.2f})"
    )

    fig2, ax2 = plt.subplots(figsize=(6,6))

    ax2.imshow(
        image,
        origin="lower",
        cmap="gray",
        vmin=np.percentile(image,5),
        vmax=np.percentile(image,99)
    )

    ax2.scatter(
        center_x,
        center_y,
        color="red",
        s=80,
        label="Center"
    )

    ax2.legend()

    st.pyplot(fig2)

    st.write("---")

    st.subheader("Surface Brightness Profile")

    radii, brightness = radial_surface_brightness(
        image,
        (center_y, center_x)
    )

    fig3, ax3 = plt.subplots(figsize=(7,5))

    ax3.plot(
        radii,
        brightness,
        color="blue",
        linewidth=2
    )

    ax3.set_xlabel("Radius (pixels)")
    ax3.set_ylabel("Surface Brightness")
    ax3.grid(True)

    st.pyplot(fig3)

    st.write("---")

    st.subheader("Effective Radius")

    re, total_flux = effective_radius(
        image,
        (center_y, center_x)
    )

    st.metric(
        "Effective Radius",
        f"{re:.2f} pixels"
    )

    st.metric(
        "Total Flux",
        f"{total_flux:.2f}"
    )

    st.write("---")

    st.subheader("Effective Radius Overlay")

    fig4, ax4 = plt.subplots(figsize=(6,6))

    ax4.imshow(
        image,
        origin="lower",
        cmap="gray",
        vmin=np.percentile(image,5),
        vmax=np.percentile(image,99)
    )

    circle = plt.Circle(
        (center_x, center_y),
        re,
        color="red",
        fill=False,
        linewidth=2
    )

    ax4.add_patch(circle)

    ax4.scatter(
        center_x,
        center_y,
        color="yellow",
        s=60
    )

    st.pyplot(fig4)

st.sidebar.title("About")

st.sidebar.info(
"""
Galaxy Structure Analyzer

Functions

• FITS Upload

• Galaxy Center Detection

• Surface Brightness Profile

• Effective Radius

Created with

Python

Streamlit

Astropy

Photutils
"""
)
import numpy as np
import pandas as pd


def save_results(radius, brightness, filename="surface_profile.csv"):
    df = pd.DataFrame({
        "Radius (pixel)": radius,
        "Surface Brightness": brightness
    })

    df.to_csv(filename, index=False)

    return filename


def galaxy_summary(center_x, center_y, effective_radius, total_flux):

    result = {
        "Center X": center_x,
        "Center Y": center_y,
        "Effective Radius": effective_radius,
        "Total Flux": total_flux
    }

    return result
    import numpy as np
from astropy.visualization import ZScaleInterval


def normalize_image(image):

    image = np.nan_to_num(image)

    interval = ZScaleInterval()

    vmin, vmax = interval.get_limits(image)

    return image, vmin, vmax


def image_statistics(image):

    return {
        "Mean": np.mean(image),
        "Median": np.median(image),
        "Maximum": np.max(image),
        "Minimum": np.min(image),
        "Std": np.std(image)
    }

     

