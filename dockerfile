# Use an official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install ODBC and dependencies for pyodbc
# Install ODBC, SSL certificates, and dependencies for pyodbc
RUN apt-get update && apt-get install -y \
    unixodbc \
    unixodbc-dev \
    libodbc1 \
    libgssapi-krb5-2 \
    ca-certificates \
    && apt-get clean && update-ca-certificates


# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project to the container's working directory
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
