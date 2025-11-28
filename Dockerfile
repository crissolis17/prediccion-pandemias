# Usa la imagen base de Airflow que ya estás usando en docker-compose
FROM apache/airflow:2.7.3-python3.11

# Instala las dependencias que Kedro necesita
# 1. Copia el archivo requirements.txt dentro del contenedor
COPY requirements.txt /tmp/requirements.txt

# 2. Instala las dependencias y limpia el cache de pip
# La ruta del proyecto Kedro es /opt/airflow/kedro-project, por lo que las
# dependencias son necesarias para que kedro funcione.
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Mantenemos el punto de entrada de Airflow
ENTRYPOINT ["/entrypoint"]
CMD ["--help"]