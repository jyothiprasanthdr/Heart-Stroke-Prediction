from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import pickle
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app =  FastAPI(title="Stroke Prediction API", description="API for predicting stroke risk using XGBoost model", version="1.0.0")

try:
    with open("Health_Stroke_Pred/stroke_xgb_deploy.pkl", "rb") as f:
        model_pipeline = pickle.load(f)
        logger.info("Model pipeline loaded successfully.")
except FileNotFoundError:
    logger.error("Model file 'Health_Stroke_Pred/stroke_xgb_deploy.pkl' not found.")
    raise Exception("Model file 'stroke_xgb_deploy.pkl' not found. Ensure it is in the same directory as the app.")


class StrokeInput(BaseModel):
    age: float
    high_glucose_flag: int
    bmi: float
    smoking_status: str
    work_type: str

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get('/', response_class=HTMLResponse)
async def root():
    logger.info("Root endpoint accessed.")
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Stroke Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 600px; margin: auto; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; }
            input, select { width: 100%; padding: 8px; margin-bottom: 10px; }
            button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; cursor: pointer; }
            button:hover { background-color: #45a049; }
            #result { margin-top: 20px; padding: 10px; border: 1px solid #ddd; display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Stroke Risk Prediction</h1>
            <form id="strokeForm">
                <div class="form-group">
                    <label for="age">Age:</label>
                    <input type="number" id="age" name="age" step="0.1" required>
                </div>
                <div class="form-group">
                    <label for="high_glucose_flag">High Glucose (0 or 1):</label>
                    <input type="number" id="high_glucose_flag" name="high_glucose_flag" min="0" max="1" required>
                </div>
                <div class="form-group">
                    <label for="bmi">BMI:</label>
                    <input type="number" id="bmi" name="bmi" step="0.1" required>
                </div>
                <div class="form-group">
                    <label for="smoking_status">Smoking Status:</label>
                    <select id="smoking_status" name="smoking_status" required>
                        <option value="never smoked">Never Smoked</option>
                        <option value="formerly smoked">Formerly Smoked</option>
                        <option value="smokes">Smokes</option>
               
                    </select>
                </div>
                <div class="form-group">
                    <label for="work_type">Work Type:</label>
                    <select id="work_type" name="work_type" required>
                        <option value="Private">Private</option>
                        <option value="Self-employed">Self-employed</option>
                        <option value="Govt_job">Government Job</option>
                        <option value="children">Children</option>
                        <option value="Never_worked">Never Worked</option>
                    </select>
                </div>
             
                <button type="submit">Predict</button>
            </form>
            <div id="result"></div>
        </div>
        <script>
            document.getElementById('strokeForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const formData = {
                    age: parseFloat(document.getElementById('age').value),
                    high_glucose_flag: parseInt(document.getElementById('high_glucose_flag').value),
                    bmi: parseFloat(document.getElementById('bmi').value),
                    smoking_status: document.getElementById('smoking_status').value,
                    work_type: document.getElementById('work_type').value,
                  
                };
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(formData)
                    });
                    const result = await response.json();
                    if (response.ok) {
                        document.getElementById('result').innerText = `Prediction: ${result.prediction}`;
                        document.getElementById('result').style.display = 'block';
                    } else {
                        document.getElementById('result').innerText = `Error: ${result.detail}`;
                        document.getElementById('result').style.display = 'block';
                    }
                } catch (error) {
                    document.getElementById('result').innerText = `Error: ${error.message}`;
                    document.getElementById('result').style.display = 'block';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post('/predict')
async def predict_stroke(input_data: StrokeInput):
    try:

        input_dict = input_data.dict(exclude_none=True)  # Exclude None values (e.g., if ever_married is not provided)
        input_df = pd.DataFrame([input_dict])
        logger.info(f"Received input: {input_dict}")

        # Validate input data
        if input_data.age < 0:
            raise HTTPException(status_code=400, detail="Age must be non-negative.")
        if input_data.high_glucose_flag not in [0, 1]:
            raise HTTPException(status_code=400, detail="high_glucose_flag must be 0 or 1.")
        if input_data.bmi <= 0:
            raise HTTPException(status_code=400, detail="BMI must be positive.")
        valid_smoking_status = ["never smoked", "formerly smoked", "smokes", "Unknown"]  # Adjust based on training data
        if input_data.smoking_status not in valid_smoking_status:
            raise HTTPException(status_code=400, detail=f"smoking_status must be one of {valid_smoking_status}.")
        valid_work_type = ["Private", "Self-employed", "Govt_job", "children", "Never_worked"]  # Adjust based on training data
        if input_data.work_type not in valid_work_type:
            raise HTTPException(status_code=400, detail=f"work_type must be one of {valid_work_type}.")



        prediction = model_pipeline.predict(input_df)[0]

                # Return result
        return {
                "prediction": "Stroke" if prediction == 1 else "No Stroke",
                "status": "success",
                "input": input_dict
            }
    except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

        # Run the app (for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)