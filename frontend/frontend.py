import io
import requests
from PIL import Image   
import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder


backend_url = "http://fastapi:8000/"

def process(image, server_url: str):

    r = requests.post(
        server_url, files={"image": ("filename", image, "image/jpeg")}
    )

    return r

if __name__ == '__main__':
    # construct UI layout
    st.title("Unet Skin Lesion Segmentation")

    st.write(
        """Obtain semantic segmentation maps of the image in input.
            This streamlit example uses a FastAPI service as backend.
            Visit this URL at `:8000/docs` for FastAPI documentation."""
    )  # description and instructions

    input_image = st.file_uploader("insert image")  # image upload widget

    if st.button("Get segmentation map"):

        col1, col2 = st.columns(2)

        if input_image:
            segments = process(input_image, backend_url + 'segmentation')
            original_image = Image.open(input_image).convert("RGB")
            print(segments.content)
            segmented_image = Image.open(io.BytesIO(segments.content)).convert("RGB")
            col1.header("Original")
            col1.image(original_image, use_column_width=True)
            col2.header("Segmented")
            col2.image(segmented_image, use_column_width=True)
        else:
            # handle case with no image
            st.write("Insert an image!")


