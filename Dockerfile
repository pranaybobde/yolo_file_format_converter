# Base image
FROM python:3.9-slim-buster

# Install required libraries
RUN apt-get update && apt-get install &&  apt install nano -y ffmpeg libsm6 libxext6 && \
    apt-get install -y libgl1-mesa-glx gcc && \
    pip install psutil && \
    pip install cython && \
    pip install torch==1.9.0 torchvision==0.10.0 opencv-python-headless==4.5.5.64 pillow==8.3.1 redis pymongo && \
    # pip install yolov5==5.0.0
    pip install ultralytics


# Copy the script to the container
COPY . /app/.

# Set the working directory
WORKDIR /app

# CMD ["sleep", "infinity"]
CMD ["python3 convert_classes.py"]


# # Base image
# FROM python:3.9-slim-buster

# # Install required libraries
# RUN apt-get update && apt-get install -y nano ffmpeg libsm6 libxext6 libgl1-mesa-glx gcc

# # Set the working directory
# WORKDIR /app

# # Copy the script to the container
# COPY . /app

# # Install Python dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Run the script
# CMD ["sleep", "infinity"]