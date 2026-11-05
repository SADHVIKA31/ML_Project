# Feature Extraction Methodology and Specification – Person 2

**Author:** Archana Ulloju (Person 2 - Classical ML & Feature Extraction Lead)  
**Project:** Machine Learning for Robust Aquatic Biodiversity Monitoring  
**Target Integration Branch:** `develop`  
**Phase:** Phase 2 – Handcrafted Feature Extraction  

---

## 1. Motivation and Domain Relevance

Underwater fish recognition presents severe imaging challenges:
- **Wavelength-dependent light attenuation:** Red light attenuates within a few meters, causing pervasive green-blue color casting.
- **Turbidity and low contrast:** Suspended particulate matter and backscattering obscure fine anatomical boundaries.
- **Morphological diversity:** Fish classification relies heavily on body contour, fin structure, and orientation-invariant aspect ratios.

To address these domain challenges, Person 2 implemented three distinct handcrafted representations:
1. **Color Histogram (RGB + HSV):** Captures multi-band radiometric distributions.
2. **Histogram of Oriented Gradients (HOG):** Captures local edge orientations, structural gradients, and contour shape.
3. **Combined Feature Representation:** Concatenates color and structural descriptors.

---

## 2. Color Histogram Formulation

- **OpenCV Implementation:** `cv2.calcHist`.
- **Color Spaces:**
  - **RGB Space:** Captures absolute spectral intensity across Red, Green, and Blue channels.
  - **HSV Space:** Decouples chromaticity (Hue and Saturation) from luminance (Value), providing invariance to variable underwater depth and lighting shadows.
- **Binning & Normalization:**
  - $8 \times 8 \times 8$ 3D bins per color space ($512$ bins each).
  - Both spaces concatenated yield $1,024$ features.
  - $L_1$ normalization ensures $\sum |h_i| = 1.0$, preventing sample bias across variable illumination.

---

## 3. Histogram of Oriented Gradients (HOG) Formulation

- **Scikit-Image Implementation:** `skimage.feature.hog`.
- **Pre-processing:** Conversion from RGB to single-channel luminance grayscale followed by standardized bilinear resize to $224 \times 224$ px.
- **Parameters:**
  - Orientations: $9$ directional bins across $[0^\circ, 180^\circ]$.
  - Pixels per cell: $16 \times 16$ px.
  - Cells per block: $2 \times 2$ cells.
  - Block normalization: `L2-Hys` (L2 norm with clipping at $0.2$ and re-normalization) to suppress local shadow artifacts.
- **Dimensionality:**
  $$\left(\frac{224}{16} - 2 + 1\right) \times \left(\frac{224}{16} - 2 + 1\right) \times 2 \times 2 \times 9 = 13 \times 13 \times 36 = 6,084\text{ features}$$

---

## 4. Combined Representation & Data Integrity

- **Concatenation:** Feature vector $\mathbf{x}_{\text{comb}} = [\mathbf{x}_{\text{col}} \parallel \mathbf{x}_{\text{hog}}] \in \mathbb{R}^{7,108}$.
- **Alignment:** Sample IDs, video IDs, and ground truth labels are strictly matched index-by-index.
- **Caching:** Compressed `.npz` files avoid repeated computational overhead.
