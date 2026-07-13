ARG IMAGE=docker.iscinternal.com/docker-intersystems/intersystems/iris-community:2026.3.0AI.108.0
FROM $IMAGE

WORKDIR /home/irisowner/dev
COPY . .

## Embedded Python environment
ENV IRISUSERNAME="_SYSTEM"
ENV IRISPASSWORD="SYS"
ENV IRISNAMESPACE="USER"
ENV PYTHON_PATH=/usr/irissys/bin/
ENV PATH="/usr/irissys/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/irisowner/bin"
ENV PYTHONPATH="/home/irisowner/dev/src"

RUN /usr/irissys/bin/irispython -m pip install polars --break-system-packages --quiet && \
    /usr/irissys/bin/irispython -m pip install langchain_intersystems-0.0.1-py3-none-any.whl --break-system-packages --quiet && \
    /usr/irissys/bin/irispython -m pip install langchain-google-genai --break-system-packages --quiet

RUN iris start IRIS && \
    iris merge IRIS merge.cpf && \
    iris session IRIS < iris.script && \
    iris stop IRIS quietly safely