FROM python:3.11-slim

WORKDIR /app/work

COPY . /app/work
RUN pip install --upgrade pip
# TODO: Install only core dependencies for the app
RUN pip install .[ml]

ENV HOME /app/work

CMD ["python", "-m", "sculpt.app"]
