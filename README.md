# Titanic ML Inference API
A production-ready, containerized RESTful API bridging Data Science and Backend Engineering.

# Project Overview
This project demonstrates a complete End-to-End MLOps pipeline. It wraps a Scikit-learn Random Forest model into a high-performance FastAPI service, ensuring reproducibility through Docker and data observability through automated SQLite logging.

# Core Value Proposition
- Model deployment beyond Jupyter Notebooks
- Strict type validation (Pydantic) and API design patterns.
- Provide containerized solution

# Key Features
This service is designed with production standards in mind:
- ML Inference: Serves a Random Forest Classifier trained on the Titanic dataset.
- High-Performance: Built with FastAPI for asynchronous, low-latency request handling.(need explaination)
- Data Persistence: Automated SQLite logging captures every request/response for monitoring.
- Dockerized: Fully containerized environment ensures consistent deployment across any OS.
- Robust Validation: Uses Pydantic schemas to enforce strict data types and prevent runtime errors.(need explaination)

# Tech Stack
This project leverages a modern, lightweight Python stack designed for microservices.
- Language: Python
- Web Framework: FastAPI + Uvicorn
- ML Core: Scikit-learn, Pandas
- Database: SQLite
- Infrastructure: Docker

# Repository Structure
The project follows a standard microservice layout:
```markdown
ml-api-project/
├── Dockerfile              # Instructions to build the container image
├── main.py                 # Entry point for FastAPI & business logic
├── train_model.py          # Script to train and serialize the model
├── requirements.txt        # Python dependencies list
├── titanic_model.pkl       # Serialized binary model file
└── titanic_logs.db         # Database file for inference logs
```

# Quick Start Guide
1. Build the Image
   ```bash
   docker build -t titanic-api .
   ```
2. Run the Container
   ```bash
   docker run -p 8000:8000 titanic-api
   ```
3. Test the API
[Visit the auto-generated documentation](http://localhost:8000/docs)

# API Documentation
Interactive Swagger UI is available at ```/docs```.

1. Health Check
   - Method: ```GET```
   - Endpoint: ```/```
   - Returns: Service status message to verify uptime.
2. Predict Survival (Core Business Logic)
   - Method: ```POST```
   - Endpoint: ```/predict```
   - Description: Submits passenger data for inference.
Example Request:
```
{
  "pclass": 1,
  "age": 25.5,
  "fare": 50.0
}
```
Example Response:
```
{
  "survived_prediction": 1,
  "survival_probability": 0.88,
  "log_status": "Saved to Database"
}
```
3. View Logs
   - Method: ```GET```
   - Endpoint: ```/logs```
   - Description: Fetches recent inference history from SQLite for debugging.

# Roadmap & Improvements
- [] Database: Migrate from SQLite to PostgreSQL for better scalability.
- [] CI/CD: Add GitHub Actions for automated linting and testing.
- [] Monitoring: Integrate Prometheus/Grafana for metric visualization.
