# COVID 19 Detection from Chest XRay Images using CNN

This project focuses on detecting **COVID-19**, **Pneumonia**, and **Normal** cases from chest X-ray images using **Convolutional Neural Networks (CNN)** and **Transfer Learning** techniques. The primary goal is to assist medical professionals by providing a fast and reliable classification tool that can help prioritize testing and treatment in resource-limited scenarios.

##  Key Features

- Trained and evaluated multiple deep learning models (e.g., VGG19, ResNet50, NASNetLarge, Xception)
- Used **transfer learning** with fine-tuning for improved performance
- Achieved high accuracy on multi-class classification: COVID-19, Pneumonia, and Normal
- Evaluation metrics include **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrix**
- Final model saved and ready to be served through a **FastAPI** interface
- Project is well-documented and reproducible

##  Dataset

The dataset used in this project is the [**COVID‑19 Radiography Database**](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database), which contains chest X-ray images for COVID‑19, Viral Pneumonia, and Normal cases. The data is publicly available on Kaggle.  

##  Results

- The best-performing model (RESNET152V2) reached:
  - **Accuracy**: ~0.9700 %
  - **Precision**: ~0.9642 %
  - **Recall**: ~ 0.9450 %
  - **F1-Score**: ~0.9545 %



## 📄 Report

A full **project report** is included in this repository, detailing the methodology, dataset preparation, model architecture, training process, and result analysis. You can find it under `report/`.


##  Running the API ( In /covid_diagnosis_project_API)

After training the model (or /covid_diagnosis_project_API/models already has trained model) in  you can run an API to serve predictions using **FastAPI**.

### ▶ Steps to Run the API

1. Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the API server:
   ```bash
   uvicorn app.main:app --reload
   ```

   This will start the API at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

3. Access the interactive API documentation (Swagger UI) at:
   ```
   http://127.0.0.1:8000/docs
   ```

4. Use the `/predict/` endpoint to send a chest X-ray image and receive prediction results (COVID-19 / Pneumonia / Normal).

