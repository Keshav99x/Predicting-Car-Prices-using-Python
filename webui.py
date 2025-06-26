import gradio as gr
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from main import regressor

# Use your trained model here
# Make sure regressor is already trained (as in your notebook)
# regressor = RandomForestRegressor(...)
# regressor.fit(X_train, Y_train)

# Prediction function
def predict_price(leather_interior, fuel_type, engine_volume, mileage,
                  cylinders, gear_box_type, drive_wheels, airbags, car_age):
    
    features = np.array([[leather_interior, fuel_type, engine_volume, mileage,
                          cylinders, gear_box_type, drive_wheels, airbags, car_age]])
    
    
    prediction = regressor.predict(features)[0]
    return f"Predicted Car Price: ${int(prediction):,}"

# Gradio interface
iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Radio([0, 1], label="Leather Interior (0=No, 1=Yes)"),
        gr.Dropdown([0, 1, 2], label="Fuel Type (0=Hybrid, 1=Petrol, 2=Diesel)"),
        gr.Number(label="Engine Volume (e.g. 2.0)"),
        gr.Number(label="Mileage (in km)"),
        gr.Number(label="Number of Cylinders"),
        gr.Dropdown([0, 1, 2, 3], label="Gear Box Type (e.g. 0=Auto, 1=Tiptronic, 2=Manual, 3=Variator)"),
        gr.Dropdown([0, 1, 2], label="Drive Wheels (0=Front, 1=4x4, 2=Rear)"),
        gr.Number(label="Airbags"),
        gr.Number(label="Car Age (in years)"),
    ],
    outputs="text",
    title="Car Price Prediction",
    description="Enter car features to estimate its market price"
)

iface.launch()