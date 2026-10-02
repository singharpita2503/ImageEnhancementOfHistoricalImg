import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Historical Photograph Enhancement",
    page_icon="📷",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #aaaaaa;
    margin-bottom: 35px;
}

.result-title {
    text-align: center;
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 10px;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #171b22;
    border: 1px solid #30363d;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.markdown(
    '<div class="title">📷 Historical Photograph Enhancement</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Digital Image Processing Based Image Enhancement'
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an old photograph",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# PROCESS IMAGE
# --------------------------------------------------

if uploaded_file is not None:

    # Read uploaded image
    image_bytes = uploaded_file.read()

    # Convert bytes to OpenCV image
    image_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Convert BGR → RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------
    # 1. GRAYSCALE CONVERSION
    # --------------------------------------------------

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --------------------------------------------------
    # 2. MEDIAN FILTER
    # --------------------------------------------------

    # Removes small noise and dust particles
    denoised_img = cv2.medianBlur(gray_img, 3)

    # --------------------------------------------------
    # 3. HISTOGRAM EQUALIZATION
    # --------------------------------------------------

    # Equivalent to MATLAB:
    # enhanced_img = histeq(gray_img);

    hist_equalized = cv2.equalizeHist(denoised_img)

    # --------------------------------------------------
    # 4. CLAHE
    # --------------------------------------------------

    # Improves local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    clahe_img = clahe.apply(hist_equalized)

    # --------------------------------------------------
    # 5. IMAGE SHARPENING
    # --------------------------------------------------

    # Gaussian blur
    blurred = cv2.GaussianBlur(
        clahe_img,
        (0, 0),
        2
    )

    # Unsharp masking
    sharpened_img = cv2.addWeighted(
        clahe_img,
        1.5,
        blurred,
        -0.5,
        0
    )

    # --------------------------------------------------
    # DISPLAY ORIGINAL + ENHANCED
    # --------------------------------------------------

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="result-title">Original Photograph</div>',
            unsafe_allow_html=True
        )

        st.image(
            img_rgb,
            use_container_width=True
        )

    with col2:

        st.markdown(
            '<div class="result-title">Enhanced Photograph</div>',
            unsafe_allow_html=True
        )

        st.image(
            sharpened_img,
            use_container_width=True,
            clamp=True
        )

    # --------------------------------------------------
    # INTERMEDIATE RESULTS
    # --------------------------------------------------

    st.markdown("---")

    st.subheader("Image Processing Stages")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### Grayscale")

        st.image(
            gray_img,
            use_container_width=True,
            clamp=True
        )

    with col2:

        st.markdown("### Histogram Equalization")

        st.image(
            hist_equalized,
            use_container_width=True,
            clamp=True
        )

    with col3:

        st.markdown("### Final Enhancement")

        st.image(
            sharpened_img,
            use_container_width=True,
            clamp=True
        )

    # --------------------------------------------------
    # HISTOGRAM COMPARISON
    # --------------------------------------------------

    st.markdown("---")

    st.subheader("Histogram Analysis")

    hist_original = cv2.calcHist(
        [gray_img],
        [0],
        None,
        [256],
        [0, 256]
    )

    hist_enhanced = cv2.calcHist(
        [sharpened_img],
        [0],
        None,
        [256],
        [0, 256]
    )

    # Normalize histogram
    hist_original = cv2.normalize(
        hist_original,
        hist_original
    ).flatten()

    hist_enhanced = cv2.normalize(
        hist_enhanced,
        hist_enhanced
    ).flatten()

    histogram_data = {
        "Original": hist_original,
        "Enhanced": hist_enhanced
    }

    st.line_chart(histogram_data)

    # --------------------------------------------------
    # DOWNLOAD BUTTON
    # --------------------------------------------------

    st.markdown("---")

    # Convert enhanced image to PNG
    success, encoded_image = cv2.imencode(
        ".png",
        sharpened_img
    )

    if success:

        st.download_button(
            label="⬇️ Download Enhanced Image",
            data=encoded_image.tobytes(),
            file_name="enhanced_historical_photo.png",
            mime="image/png"
        )

    # --------------------------------------------------
    # INFORMATION
    # --------------------------------------------------

    st.markdown("""
    <div class="info-box">

    <b>Processing Techniques Used:</b>

    <br><br>

    1. Grayscale Conversion  
    <br>
    2. Median Filtering  
    <br>
    3. Histogram Equalization  
    <br>
    4. CLAHE (Local Contrast Enhancement)  
    <br>
    5. Unsharp Masking (Image Sharpening)

    </div>
    """, unsafe_allow_html=True)

else:

    st.info(
        "Upload an old photograph above to begin enhancement."
    )