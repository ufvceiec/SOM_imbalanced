FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install jupyter

COPY requirements.txt ./

RUN apt-get update && apt-get install -y git
RUN pip install -r requirements.txt