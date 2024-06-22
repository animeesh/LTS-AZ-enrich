FROM python:3.9-slim
WORKDIR /ver2

RUN apt-get update \
    && apt-get install -y --no-install-recommends apt-utils libgl1 libglib2.0-0 \
    python3-pip \
    && apt-get install libzbar-dev -y \
    && apt-get clean \
    && apt-get autoremove



COPY requirements.txt .
COPY . .
RUN pip3 install --upgrade pip
RUN pip install -r requirements.txt
RUN pip3 install cmake lit
RUN pip3 install typing-extensions --upgrade

RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org python-multipart
# EXPOSE 8000
CMD ["python", "app.py"]

