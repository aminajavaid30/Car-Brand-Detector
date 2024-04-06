import streamlit as st
from PIL import Image
import os
import shutil
from ultralytics import YOLO

# 1. Welcome message and brief description
st.title("Car Brand Detector")
st.subheader("Upload the image of a car (or multiple cars) and we will detect the brand!")

# 2. Initialize YOLO with our custom trained model name
model = YOLO("best.pt")

# 3. Add a file uploader widget to the sidebar
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
uploaded_image_path = ''

# Check if a file was uploaded
if uploaded_file is not None:

    col1, col2 = st.columns((1, 2))

    with col1:
        # Display the uploaded image
        image = Image.open(uploaded_file)

        # Calculate the new width based on the aspect ratio
        aspect_ratio = image.width / image.height
        new_width = int(400 * aspect_ratio)
        
        # Resize the image while maintaining aspect ratio
        image_resized = image.resize((new_width, 400))

        st.image(image_resized, caption='Uploaded Image')

        # Save the uploaded file to a directory named 'uploaded_files'
        save_dir = 'uploaded_files'
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
        file_path = os.path.join(save_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())

        uploaded_image_path = file_path

    with col2:
        # Create a button for brand detection
        if st.button('Detect Brand'):
            # Perform brand detection when the button is clicked
            # Use the pre-trained YOLO model to detect the car brand from the uploaded image
            results = model.predict(source=uploaded_image_path, save=True, save_txt=True)
            
            # Display the uploaded image
            image = Image.open('runs/detect/predict/'+uploaded_file.name)

            # Calculate the new width based on the aspect ratio
            aspect_ratio = image.width / image.height
            new_width = int(400 * aspect_ratio)

            # Resize the image while maintaining aspect ratio
            image_resized = image.resize((new_width, 400))

            classes = []
            for detection in results:
                # Assuming detection.boxes.cls and detection.boxes.conf are tensors with multiple elements
                for class_id_tensor, confidence_tensor in zip(detection.boxes.cls, detection.boxes.conf):
                    class_id = int(class_id_tensor.item())
                    confidence = round(float(confidence_tensor.item()), 2)
                    class_name = results[0].names[class_id].capitalize()
                    classes.append((class_name, confidence))
                
            caption = 'Brand(s) detected: '+str(classes)     
            st.image(image_resized, caption=caption)

            for c in classes:
                st.write("Brand:", c[0],"--- Confidence:", c[1])
    
            # Delete the 'uploaded_files' directory after brand detection
            if os.path.exists('uploaded_files'):
                shutil.rmtree('uploaded_files')

            # Delete the 'runs' directory after brand detection
            if os.path.exists('runs'):
                shutil.rmtree('runs')