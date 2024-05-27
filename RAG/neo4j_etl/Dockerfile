FROM python:3.10-slim

WORKDIR /app

COPY ./src/ /app/

COPY ./pyproject.toml /code/pyproject.toml
RUN pip install /code/.

CMD ["sh", "entrypoint.sh"]