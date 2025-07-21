

**Crop Recommendation System**

This machine learning project predicts the **most suitable crop** to grow based on environmental and soil conditions such as **NPK values**, **temperature**, **humidity**, **pH**, and **rainfall**.

---

**Dataset**

We used the [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) from Kaggle, which contains over 2200 entries with the following features:

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* pH
* Rainfall
* Crop label (target)

---

**Project Structure**

```
crop-recommendation/
├── app.py                 --> Streamlit web app
├── crop.csv               --> Dataset
├── model.pkl              --> Trained Random Forest model
├── notebook.ipynb         --> EDA and model training
├── requirements.txt       --> List of dependencies
└── README.md              --> Project documentation
```

---

**Key Features**

* Preprocessing using Label Encoding
* Model building using:
  • Decision Tree
  • Random Forest
* Visualizations:
  • Decision Tree Plot
  • Random Forest Feature Importances
* User-friendly interface using Streamlit

---

**Visualizations Included**

* Decision Tree visualized using `plot_tree`
* Feature importance plotted with Seaborn

---

**How to Run**

1. Clone the repository:

   ```
   git clone https://github.com/sanyagupta31/crop-recommendation.git
   cd crop-recommendation
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Run the app:

   ```
   streamlit run app.py
   ```

---

**Tools and Libraries Used**

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit
* Matplotlib, Seaborn

---

**Created By**

* Sanya Gupta


---

**Purpose**

This project was developed to apply machine learning classification algorithms and deploy them in a real-time recommendation setting using a web interface.

---

**License**

Open for educational and demonstration purposes.

