# Use official Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy everything from root to /app (ignoring things in .dockerignore)
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose ports for FastAPI and Streamlit
EXPOSE  8501

# Make entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Start both backend and frontend when container starts
CMD ["/app/entrypoint.sh"]