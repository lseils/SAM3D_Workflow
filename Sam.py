from ultralytics import SAM

# Load the SAM 3 model
model = SAM("sam3_base.pt") # Ultralytics will auto-download the weights

# You can use text prompts in SAM 3 to isolate the building!
results = model.predict(
    "street_images/facade_001.jpg", 
    prompt="building facade", # Positive prompt
    save=True
)

# Place Holder for more code to come in the future!
