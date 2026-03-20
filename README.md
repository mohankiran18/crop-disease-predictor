# Crop Disease Prediction System

Early detection of crop diseases from leaf images using a Convolutional Neural Network, served through an interactive web interface.

---

## Problem Statement

Crop diseases cause significant agricultural losses each year. Diagnosing them accurately requires domain expertise and manual inspection — resources most smallholder farmers lack. Delayed or incorrect identification leads to widespread crop damage that could have been prevented with timely intervention.

---

## Solution Overview

This system enables farmers and agronomists to upload a leaf image and receive an instant disease classification. A trained CNN model processes the image through a Streamlit-based interface, returning the predicted disease class without requiring any expert intervention on the user's end.

---

## Key Features

- Classifies plant diseases directly from leaf images
- Delivers predictions in real time through a browser-based interface
- Preprocessing pipeline handles image resizing and normalization automatically
- Modular codebase separating model logic from the UI layer
- Runs locally with minimal setup; extensible for cloud deployment

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Deep Learning | TensorFlow, Keras |
| Image Processing | OpenCV |
| Data Handling | NumPy, Pandas |
| Evaluation | Scikit-learn, Matplotlib |
| Interface | Streamlit |
| Version Control | Git |

---

## Model and Approach

- **Architecture**: Custom CNN with convolutional, pooling, and fully connected layers trained for multi-class image classification
- **Preprocessing**: Images are resized to a fixed input dimension and pixel values normalized to [0, 1]
- **Dataset**: Trained on a labeled plant disease dataset covering multiple crop species and disease categories
- **Evaluation**: Model performance assessed using classification accuracy, confusion matrix, and per-class precision/recall

---

## Project Architecture

```
Input Image
    │
    ▼
Preprocessing (resize, normalize)
    │
    ▼
CNN Model (feature extraction + classification)
    │
    ▼
Predicted Disease Class + Confidence Score
    │
    ▼
Streamlit UI Output
```

---

## Folder Structure

```
crop-disease-predictor/
│
├── Backend_Model/          # Model definition, training scripts, and inference logic
├── UI/                     # Streamlit application and frontend components
├── plant_disease/          # Dataset directory and image handling utilities
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## Installation

**Prerequisites**: Python 3.8 or higher

1. Clone the repository:

```bash
git clone https://github.com/mohankiran18/crop-disease-predictor.git
cd crop-disease-predictor
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Launch the application:

```bash
streamlit run app.py
```

Once running:

1. Open the URL shown in your terminal (typically `http://localhost:8501`)
2. Upload a clear image of a plant leaf
3. View the predicted disease class and confidence output

---

## Results

| Metric | Value |
|---|---|
| Classification Accuracy | _To be updated after final evaluation_ |
| Inference Time (avg) | _To be updated_ |
| Number of Classes | _To be updated_ |

> Results will be populated following final model evaluation on the held-out test set.

---

## Future Improvements

- Expand the dataset to cover additional crop species and disease variants
- Experiment with transfer learning (e.g., EfficientNet, ResNet) to improve classification accuracy
- Package the model as a REST API for integration with third-party applications
- Deploy to a cloud platform (AWS, GCP, or HuggingFace Spaces) for public access
- Build a lightweight mobile interface for field use

---

## Author

**Mohan Kiran**  
GitHub: [github.com/mohankiran18](https://github.com/mohankiran18)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "Add: brief description"`
4. Push the branch: `git push origin feature/your-feature-name`
5. Open a pull request with a clear description of what you changed and why
