FROM apache/airflow:3.1.8
USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir \
    --constraint https://raw.githubusercontent.com/apache/airflow/constraints-3.1.8/constraints-3.12.txt \
    -r /requirements.txt