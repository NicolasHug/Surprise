FROM python:3.6
MAINTAINER Marc Feger <marc.feger@uni-duesseldorf.de>

RUN apt-get update -yqq &&\
    apt-get upgrade -yqq &&\
    apt-get install gcc libenchant1c2a -yqq --fix-missing

COPY . /surprise
WORKDIR surprise

RUN pip install -Uq pip &&\
    pip --quiet install -r requirements_dev.txt && \
    pip --quiet install scikit-surprise && \
    pip install -e .

RUN ["python", "setup.py", "install"]
