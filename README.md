I serve an Unet model for skin lesion segmentation using `FastAPI` for the backend service and `streamlit` for the frontend service. `docker-compose` orchestrates the two services and allows communication between them.

To run the example in a machine running Docker and docker-compose, run:
    docker-compose build
    docker-compose up

To visit the FastAPI documentation of the resulting service, visit http://localhost:8000 with a web browser.  
To visit the streamlit UI, visit http://localhost:8501.
