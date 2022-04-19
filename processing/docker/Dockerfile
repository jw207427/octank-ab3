
FROM public.ecr.aws/docker/library/python:3.7
    
ADD requirements.txt /
ADD video_processing.py /
ADD octank_utility.py /

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install -r requirements.txt

ENV PYTHONUNBUFFERED=TRUE 
ENV TF_CPP_MIN_LOG_LEVEL="2"

ENTRYPOINT ["python3", "/video_processing.py"]
