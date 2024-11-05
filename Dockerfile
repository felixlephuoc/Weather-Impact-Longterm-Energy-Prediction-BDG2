# Use an official Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose the port the app will run on
EXPOSE 8000

# Run the Flask application
CMD ["python", "src/main.py"]
