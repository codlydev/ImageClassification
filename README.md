Image Classifier using TensorFlow & OpenCV
This project is a deep learning-based image classifier that predicts objects using a pre-trained CNN model (CIFAR-10 dataset). It utilizes:
✅ TensorFlow/Keras for model loading and prediction
✅ OpenCV for image preprocessing
✅ NumPy for data manipulation

🔹 Features:
✔ Loads and preprocesses images (resizing, normalization)
✔ Uses a trained CNN model to classify images into 10 categories (Airplane, Dog, Cat, etc.)
✔ Predicts class and outputs the result

🔹 Setup & Usage:

bash
Copy
Edit
pip install tensorflow opencv-python numpy
python main.py
💡 Ensure the trained model (image_classifier.keras) and test images are in the correct directory!
