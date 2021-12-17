FROM python:3.9

RUN apt-get update
RUN apt-get install -y llvm

COPY requirements.txt .

RUN python3 -m pip install -r requirements.txt

ENTRYPOINT [ "/bin/bash" ]