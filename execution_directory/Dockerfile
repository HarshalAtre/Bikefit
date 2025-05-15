# Use Python 3.10 slim base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Create uploads folder (optional safeguard)
RUN mkdir -p ./uploads

# Expose port expected by Hugging Face
EXPOSE 7860

# Start the Flask app on correct port using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
