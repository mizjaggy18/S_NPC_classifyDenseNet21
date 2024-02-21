FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu18.04
CMD nvidia-smi

FROM cytomine/software-python3-base:v2.2.0

#INSTALL
RUN pip install numpy
RUN pip install shapely
RUN pip install torch
RUN pip install torchvision

RUN mkdir -p /models 
ADD 66k-4classnpc_densenet21adamssv_best_model_100ep.pth /models/66k-4classnpc_densenet21adamssv_best_model_100ep.pth
RUN chmod 444 /models/66k-4classnpc_densenet21adamssv_best_model_100ep.pth

#ADD FILES
RUN mkdir -p /app
ADD descriptor.json /app/descriptor.json
ADD run.py /app/run.py


ENTRYPOINT ["python3", "/app/run.py"]
