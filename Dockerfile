# Use the official slim Python image matching the project's Python version
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the API code and the trained model into the container
COPY app.py .
COPY models/churn_model.pkl models/churn_model.pkl

# Document the port the API listens on
EXPOSE 8000

# Start the API; 0.0.0.0 lets requests from outside the container reach it
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]