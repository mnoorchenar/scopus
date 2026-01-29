FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Run Flask app
CMD ["python", "app.py"]
