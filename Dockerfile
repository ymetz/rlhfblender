ARG PARENT_IMAGE
FROM $PARENT_IMAGE
ARG PYTORCH_DEPS=cpuonly
ARG PYTHON_VERSION=3.10
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)

# Install OpenCV to encode video
#USER root
#RUN apt-get update && apt-get install -y ffmpeg
#USER $MAMBA_USER

# Install micromamba env and dependencies
RUN micromamba install -n base -y python=$PYTHON_VERSION \
    pytorch $PYTORCH_DEPS opencv -c conda-forge -c pytorch -c nvidia && \
    micromamba clean --all --yes

ENV CODE_DIR /home/$MAMBA_USER

# Copy setup file only to install dependencies
COPY --chown=$MAMBA_USER:$MAMBA_USER ./setup.py ${CODE_DIR}/rlhfblender/setup.py
COPY --chown=$MAMBA_USER:$MAMBA_USER ./rlhfblender/version.txt ${CODE_DIR}/rlhfblender/rlhfblender/version.txt
COPY --chown=$MAMBA_USER:$MAMBA_USER ./rlhfblender/ ${CODE_DIR}/rlhfblender/rlhfblender/
COPY --chown=$MAMBA_USER:$MAMBA_USER ./configs/ ${CODE_DIR}/rlhfblender/configs/
COPY --chown=$MAMBA_USER:$MAMBA_USER ./rlhfblender_demo_models/ ${CODE_DIR}/rlhfblender/rlhfblender_demo_models/

RUN cd ${CODE_DIR}/rlhfblender && \
    pip install -e .[tests,docs] && \
    # Use headless version for docker
    #pip uninstall -y opencv-python && \
    pip install opencv-python-headless && \
    pip cache purge

WORKDIR ${CODE_DIR}/rlhfblender

CMD python rlhfblender/app.py