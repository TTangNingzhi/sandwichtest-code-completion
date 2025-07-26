FROM python:3.11.13-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir jsonlines tqdm
ENTRYPOINT ["python"]
