from math import ceil
import fastapi
import joblib
import numpy as np

app = fastapi.FastAPI()

classifier = joblib.load("classifier_model.pkl")
regressor = joblib.load("final_xgboost_model.pkl")

@app.get("/")
def read_root():
    return {"message": "Welcome to the The Next Pit Stop Predictor!"}


@app.post("/predict")
def predict(data: dict):
    """
    Predicts the next pit stop based on the input data.

    Args:
        data (dict): A dictionary containing the following keys:
            - TyreLife                 
            - LapTime_label            
            - Compound_label           
            - PitInTime_label          
            - TrackTemp                
            - Humidity                 
            - Rainfall_encoded         
            - AirTemp                  
            - Sector3Time_label        
            - LapNumber                
            - Sector2Time_label        
            - Sector1Time_label        
            - Position                 
            - PitOutTime_label         
            - FreshTyre_encoded        
    """
    try:
        # Extract features from the input data
        features = np.array([
            data.get("TyreLife", 0),
            data.get("LapTime_label", 0),
            data.get("Compound_label", 0),
            data.get("PitInTime_label", 0),
            data.get("TrackTemp", 0.0),
            data.get("Humidity", 0.0),
            data.get("Rainfall_encoded", 0.0),
            data.get("AirTemp", 0.0),
            data.get("Sector3Time_label", 0),
            data.get("LapNumber", 0),
            data.get("Sector2Time_label", 0),
            data.get("Sector1Time_label", 0),
            data.get("Position", 0),
            data.get("PitOutTime_label", 0),
            data.get("FreshTyre_encoded", 0)
        ]).reshape(1, -1)

        # Make predictions using the classifier and regressor
        will_pit = classifier.predict(features)[0]

        if will_pit == 1:
        # Step 2: Predict how many laps until the next pit stop
            laps_left = regressor.predict(features)[0]
            return {
            "pit_stop_prediction": int(will_pit),
            "lap_time_prediction": int(ceil(laps_left))
            }

        else:

            return {
                "pit_stop_prediction": int(will_pit),
                "lap_time_prediction": 0
            }
    except Exception as e:
        return {"error": str(e)}



