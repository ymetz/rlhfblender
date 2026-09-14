ARG PARENT_IMAGE
FROM $PARENT_IMAGE
ARG PYTORCH_DEPS=cpuonly
ARG PYTHON_VERSION=3.10
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)

# Install micromamba env and dependencies
RUN micromamba install -n base -y python=$PYTHON_VERSION \
    pytorch $PYTORCH_DEPS opencv -c conda-forge -c pytorch -c nvidia && \
    micromamba install -c conda-forge ffmpeg glew mesalib glfw && \
    micromamba clean --all --yes

# MuJoCo's OSMesa backend needs a shared library discoverable by the system loader.
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends libosmesa6 && \
    rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

ENV CODE_DIR=/home/${MAMBA_USER}
ENV DISPLAY=:99
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
# Disable numba JIT compilation as it causes problems in the docker container
ENV NUMBA_DISABLE_JIT=1

# Copy setup file only to install dependencies
COPY --chown=$MAMBA_USER:$MAMBA_USER ./setup.py ${CODE_DIR}/rlhfblender/setup.py
COPY --chown=$MAMBA_USER:$MAMBA_USER ./rlhfblender/version.txt ${CODE_DIR}/rlhfblender/rlhfblender/version.txt
COPY --chown=$MAMBA_USER:$MAMBA_USER ./rlhfblender/ ${CODE_DIR}/rlhfblender/rlhfblender/
COPY --chown=$MAMBA_USER:$MAMBA_USER ./multi-type-feedback/ ${CODE_DIR}/rlhfblender/multi-type-feedback/
COPY --chown=$MAMBA_USER:$MAMBA_USER ./configs/ ${CODE_DIR}/rlhfblender/configs/
# Runtime datasets, databases and models are supplied through bind mounts.

RUN cd ${CODE_DIR}/rlhfblender && \
    # Require a prebuilt PyAV wheel instead of compiling against system FFmpeg.
    pip install --only-binary=av -e . && \
    # Use headless version for docker
    #pip uninstall -y opencv-python && \
    pip install opencv-python-headless && \
    pip cache purge

# Fail the build if headless rendering cannot load its native libraries.
RUN python -c 'import mujoco; model = mujoco.MjModel.from_xml_string("<mujoco/>"); data = mujoco.MjData(model); renderer = mujoco.Renderer(model, height=32, width=32); renderer.update_scene(data); assert renderer.render().shape == (32, 32, 3); renderer.close()'

WORKDIR ${CODE_DIR}/rlhfblender

CMD python rlhfblender/app.py
