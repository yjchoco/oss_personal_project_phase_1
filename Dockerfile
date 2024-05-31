FROM python:3.9


WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir /workspace/
COPY . /workspace/

WORKDIR /workspace

ENTRYPOINT ["uvicorn"]
CMD ["--host=0.0.0.0", "--port=8000", "main:app"] 