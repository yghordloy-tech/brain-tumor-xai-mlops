# 1. Start with a lightweight Linux machine running Python 3.12
FROM python:3.12-slim

# 2. Set the working directory inside our container
WORKDIR /app

# 3. Copy our PRODUCTION requirements file and install the dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# 4. Copy all of our project files (API, configs, and model weights) into the container
COPY . .

# 5. Open up port 8000 so the outside world can talk to our API
EXPOSE 8000

# 6. The exact command to boot up the web server when the container turns on
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]