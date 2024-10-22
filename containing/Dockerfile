FROM mambaorg/micromamba:latest
# repo_root/
# ├ virtual_environment/
# │   ├ environment.yml 
# │   ├ requirements.txt
# │   └ ...
# ├ notebooks/
# │   └ ...
# ├ python_scripts/
# │   └ ...
# :

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN echo "patate"
RUN echo "hello there"
ARG VIRTUAL_ENVIRONMENT_FILEPATH
ARG KERNEL_NAME
ARG PORT

RUN micromamba info
RUN eval "$(micromamba shell hook --shell bash)"
COPY --chown=$MAMBA_USER:$MAMBA_USER $VIRTUAL_ENVIRONMENT_FILEPATH /tmp/env.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER ./python_modules/ ./python_modules/
COPY --chown=$MAMBA_USER:$MAMBA_USER ./data_from_internet/ ./data_from_internet/
COPY --chown=$MAMBA_USER:$MAMBA_USER ./number_from_figure/ ./number_from_figure/
COPY --chown=$MAMBA_USER:$MAMBA_USER ./faces_from_image/ ./faces_from_image/
COPY --chown=$MAMBA_USER:$MAMBA_USER ./emotion_from_face/ ./emotion_from_face/
COPY --chown=$MAMBA_USER:$MAMBA_USER ./emotions_from_faces_from_video/ ./emotions_from_faces_from_video/
RUN micromamba install --channel conda-forge --yes --name base --file /tmp/env.yaml && \
    micromamba clean --all --yes
RUN micromamba run --name=base python -c 'import uuid; print(uuid.uuid4())' > /tmp/my_uuid
RUN touch you_are_in_a_container_btw
RUN python -c 'import uuid; print(uuid.uuid4())' > /tmp/my_uuid
RUN micromamba install --channel conda-forge --yes jupyter
RUN micromamba install --channel conda-forge --yes jupyterlab
RUN micromamba install --channel conda-forge --yes nb_conda_kernels
RUN micromamba install --channel conda-forge --yes ipykernel
RUN micromamba run \
  python \
    -m ipykernel \
    install \
      --user \
      --name base_kernel \
      --display-name $KERNEL_NAME
CMD jupyter lab --no-browser --port $PORT --ip=0.0.0.0

