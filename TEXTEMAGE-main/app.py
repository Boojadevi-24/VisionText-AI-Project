import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image
import tempfile
st.set_page_config(page_title="AI OCR Tool", layout="centered")
st.title("AI-Powered Multi-Language Image-to-Text Converter")
st.write("Upload an image and extract text using a deep learning OCR model.")
langs = st.sidebar.multiselect(
    "Select Languages",
    ["en", "hi", "ta", "te", "fr", "de"],  # add as needed
    default=["en"]
)
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    img = cv2.imread(tfile.name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    st.image(thresh, caption="Processed Image", channels="GRAY")
    st.write("Extracting text...")
    reader = easyocr.Reader(langs)
    results = reader.readtext(thresh, detail=0)

    extracted_text = "\n".join(results)
    st.subheader("📄 Extracted Text")
    st.text(extracted_text)
    st.download_button("Download as TXT", extracted_text, file_name="extracted_text.txt")

