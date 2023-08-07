FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Set timezone:
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
ARG DEBIAN_FRONTEND=noninteractive

RUN  apt-get update && apt-get install -y --no-install-recommends git gcc python3.10 python3-distutils python3-pip python3-dev

# Yes, the entire following code is just to install an opencv with avc1 support, lul
RUN apt-get install -y build-essential cmake python3-numpy \
libavcodec-dev libavformat-dev libswscale-dev \
libgstreamer-plugins-base1.0-dev \
libgstreamer1.0-dev libgtk-3-dev \
libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev \
libopencv-dev x264 libx264-dev libssl-dev ffmpeg

RUN python3 -m pip install --no-binary opencv-python-headless opencv-python-headless

RUN apt-get purge -y --auto-remove && rm -rf /var/lib/apt/lists/*

# == Now we can continue with the actual dockerfile ==


# Install Python requirements
COPY requirements.txt /tmp/

RUN pip3 install --upgrade pip

# Copy the application code
COPY . .

# Only install packages after copying the code (because we do want to install local third-party packages)
RUN pip3 install -r /tmp/requirements.txt

# Make sure python output is printed directly to stdout
ENV PYTHONUNBUFFERED=1

# Add Pythonpath
ENV PYTHONPATH=/app

EXPOSE 8080 8080

# Run the application
ENTRYPOINT ["python3", "app/app.py"]
