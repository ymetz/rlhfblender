ARG PARENT_IMAGE
FROM $PARENT_IMAGE
ARG PYTORCH_DEPS=cpuonly
ARG PYTHON_VERSION=3.10
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)
ARG MUJOCO_GL_BACKEND=egl

# Install modern headless rendering libs for `mujoco` (new bindings):
# default to EGL, with OSMesa available as fallback.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libegl1 \
    libgl1 \
    libgles2 \
    libglvnd0 \
    libglfw3 \
    libosmesa6 \
    libosmesa6-dev && \
    rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

# Install micromamba env and dependencies
RUN micromamba install -n base -y python=$PYTHON_VERSION \
    pytorch $PYTORCH_DEPS -c conda-forge -c pytorch -c nvidia && \
    micromamba install -c conda-forge libgcc-ng libstdcxx-ng && \
    micromamba clean --all --yes

ENV CODE_DIR=/home/${MAMBA_USER}
ENV DISPLAY=:99
ENV MUJOCO_GL=${MUJOCO_GL_BACKEND}
ENV PYOPENGL_PLATFORM=${MUJOCO_GL_BACKEND}
ENV LD_LIBRARY_PATH=/opt/conda/lib:${LD_LIBRARY_PATH}
# Disable numba JIT compilation as it causes problems in the docker container
ENV NUMBA_DISABLE_JIT=1

# Copy setup file only to install dependencies
COPY --chown=$MAMBA_USER:$MAMBA_USER ./setup.py ${CODE_DIR}/rlhfblender/setup.py
COPY --chown=$MAMBA_USER:$MAMBA_USER ./rlhfblender/version.txt ${CODE_DIR}/rlhfblender/rlhfblender/version.txt
COPY --chown=$MAMBA_USER:$MAMBA_USER ./rlhfblender/ ${CODE_DIR}/rlhfblender/rlhfblender/
COPY --chown=$MAMBA_USER:$MAMBA_USER ./multi-type-feedback/ ${CODE_DIR}/rlhfblender/multi-type-feedback/
COPY --chown=$MAMBA_USER:$MAMBA_USER ./configs/ ${CODE_DIR}/rlhfblender/configs/
COPY --chown=$MAMBA_USER:$MAMBA_USER ./rlhfblender_model/ ${CODE_DIR}/rlhfblender/rlhfblender_model/
# Just for the study deployment, copy local data and database, so that we not need to re-generate them on the slow server
COPY --chown=$MAMBA_USER:$MAMBA_USER ./remote_data/ ${CODE_DIR}/rlhfblender/data/
COPY --chown=$MAMBA_USER:$MAMBA_USER ./remote_data/rlhfblender.db ${CODE_DIR}/rlhfblender/rlhfblender.db

RUN cd ${CODE_DIR}/rlhfblender && \
    pip install -e . && \
    # Use headless version for docker
    #pip uninstall -y opencv-python && \
    pip install opencv-python-headless && \
    pip cache purge

WORKDIR ${CODE_DIR}/rlhfblender

CMD python rlhfblender/app.py
