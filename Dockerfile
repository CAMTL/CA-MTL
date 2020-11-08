FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

ARG WORKDIR=/project

RUN mkdir -p $WORKDIR
RUN chmod o+w $WORKDIR
WORKDIR $WORKDIR
ENV HOME $WORKDIR 


RUN apt-get update --fix-missing
RUN apt-get update -y
RUN apt-get install -y \
    wget bzip2 \
    libglib2.0-0 libxext6 libsm6 libxrender1 libxml-parser-perl \
    git \
    build-essential \
    vim \
    cmake curl grep sed dpkg \
    zip unzip \
    libhdf5-dev \
    locales python-dev \
    && rm -rf /var/lib/apt/lists/*

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh
ENV PATH /opt/conda/bin:$PATH

RUN git clone https://github.com/NVIDIA/apex.git && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

COPY requirements.txt /

RUN pip install --upgrade pip
RUN pip install --requirement /requirements.txt

RUN useradd -d $HOME dockeruser
RUN chown dockeruser:dockeruser $HOME
USER dockeruser

ADD . $HOME 
