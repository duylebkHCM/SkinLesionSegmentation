import base64
import os
import uuid
import re
import io
import uvicorn
from fastapi import FastAPI, File
from starlette.responses import Response
from get_segment import get_segments, get_segmentor

model = get_segmentor()

app = FastAPI(
    title="Segmentor",
    description="Segmentor API",
    version="0.1.0",
)

@app.post("/segmentation")
def get_segmentation(image: bytes = File(...)):
    segmented_image = get_segments(model, image)
    print('Segment image', segmented_image.size)
    byte_io = io.BytesIO()
    segmented_image.save(byte_io, format='PNG')
    return Response(byte_io.getvalue(), media_type='image/png')

@app.get("/download")
def download_image(object_to_download, filename, button_text):
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    filename (str): filename and extension of file. e.g. face1.jpg
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_img, 'face1.jpg', 'Click to download image!')
    """
    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'
    return dl_link
