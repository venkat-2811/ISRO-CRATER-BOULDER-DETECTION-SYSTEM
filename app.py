import streamlit as st
import torch
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the trained YOLOv5 model
model_path = 'C:/Users/venka/OneDrive/Desktop/moon_crater_detection/scripts/yolov5_crater_detection/exp/weights/best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Team Details
st.title("Crater and Boulder Detection")
st.write("""
### Team: CraterDetectives
- **Sunkara Venkata Karthik Sai** (Team Lead)
- **Jataved Reddy**
- **M Ruhika**
""")

# Upload image section
st.header("Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Detecting craters and boulders...")

    # Run YOLOv5 model on the uploaded image
    results = model(image)

    # Display results
    results.show()
    st.image(results.render()[0], caption='Detected Image', use_column_width=True)
    
    # Save the detected image
    detected_image_path = os.path.join("yolov5_crater_detection", "detected_image.jpg")
    results.save(save_dir="yolov5_crater_detection")

    # Display coordinates of detected craters and boulders
    st.header("Detected Craters/Boulders Coordinates")
    coordinates_df = results.pandas().xyxy[0]
    st.write(coordinates_df)

    # Visualize the size distribution of detected craters/boulders
    st.header("Size Distribution of Detected Craters/Boulders")
    coordinates_df['size'] = (coordinates_df['xmax'] - coordinates_df['xmin']) * (coordinates_df['ymax'] - coordinates_df['ymin'])
    fig, ax = plt.subplots()
    sns.histplot(coordinates_df['size'], bins=20, kde=True, ax=ax)
    ax.set_title('Size Distribution of Detected Craters/Boulders')
    ax.set_xlabel('Size (pixels)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Visualize the location distribution of detected craters/boulders
    st.header("Location Distribution of Detected Craters/Boulders")
    fig = px.scatter(coordinates_df, x='xmin', y='ymin', title='Location Distribution of Detected Craters/Boulders')
    st.plotly_chart(fig)

# Display Model Accuracy
st.header("Model Accuracy")

# Read the results.csv file to get accuracy metrics
results_csv_path = 'C:/Users/venka/OneDrive/Desktop/moon_crater_detection/scripts/yolov5_crater_detection/exp/results.csv'
if os.path.exists(results_csv_path):
    results_df = pd.read_csv(results_csv_path)
    # Display column names to help identify the correct metric column
    st.write("Available columns in results.csv:")
    st.write(results_df.columns)

    # Try to find the accuracy metric
    possible_accuracy_columns = ['metrics/mAP_0.5', 'metrics/mAP_50', 'accuracy']  # Add any other possible names
    accuracy = None
    for col in possible_accuracy_columns:
        if col in results_df.columns:
            accuracy = results_df[col].iloc[-1] * 100
            break

    if accuracy is not None:
        st.write(f"Accuracy (mAP@0.5): {accuracy:.2f}%")
    else:
        st.write("Accuracy metric not found in results.csv. Please check the column names above.")
else:
    st.write("Results file not found. Please ensure the model training has been completed and the results.csv file is present.")
