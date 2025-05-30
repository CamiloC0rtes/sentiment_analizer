
---

```markdown
# 📊 Sentiment Analyzer – API & Web Interface

This project provides a **FastAPI-based sentiment analysis API** powered by a machine learning model trained on the IMDb dataset. It includes a lightweight **web interface (`Visual.html`)** for interacting with the model visually.

---

## 🚀 Features

- ✅ **REST API** for analyzing single or batch text inputs.
- 🤖 **ML model** trained using scikit-learn and saved with Joblib.
- 🧹 **Consistent preprocessing** between training and inference.
- 🌐 **Modern web interface** for quick testing.
- 🔍 **Health check** and **model info** endpoints.

---

## 🧱 Project Structure

```

analizadorsentimientos/
│
├── api/
│   ├── main.py           # FastAPI app entrypoint
│   └── predict.py        # Model loading and prediction logic
│
├── model/
│   ├── train\_model.py    # Model training script
│   └── model.pkl         # Trained model (auto-generated)
│
├── Visual.html           # Web interface for sentiment analysis
├── requirements.txt      # Project dependencies
└── README.md             # You're here

````

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/analizadorsentimientos.git
   cd analizadorsentimientos
````

2. **Create and activate a virtual environment** (optional but recommended)

   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download IMDb dataset**
   Download the `IMDB Dataset.csv` from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it in the `model/` directory.

---

## 🧠 Train the Model

1. Navigate to the `model/` folder:

   ```bash
   cd model
   ```

2. Run the training script:

   ```bash
   python train_model.py
   ```

   This will generate the file `model.pkl`.

---

## 🚀 Run the API

From the root directory of the project:

```bash
uvicorn api.main:app --reload --port 8000
```

* API will be available at [http://localhost:8000](http://localhost:8000)
* Swagger docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔌 Main Endpoints

| Method | Endpoint         | Description                          |
| ------ | ---------------- | ------------------------------------ |
| POST   | `/predict`       | Predict sentiment for a single text  |
| POST   | `/predict/batch` | Predict sentiment for multiple texts |
| GET    | `/model/info`    | Returns model information            |
| GET    | `/health`        | Health check of the API              |
| GET    | `/docs`          | Swagger UI docs                      |

---

## 🌍 Web Interface

1. Open `Visual.html` in a browser.
2. Make sure the API is running.
3. Modify the `API_BASE_URL` inside the HTML if you change the API port.

---

## 💡 Notes

* Only binary sentiment classification: **positive** or **negative**.
* Preprocessing logic must match between training and prediction.
* If predictions seem inconsistent, retrain the model and verify `model.pkl`.
* For production environments, **secure CORS settings**.

---

## 🙌 Credits

* Built with **FastAPI**, **scikit-learn**, **NLTK**, and **IMDb dataset from Kaggle**.
* Web interface built with vanilla **HTML, CSS, and JavaScript**.

---

## 🛠 Questions or Issues?

Feel free to open an [issue](https://github.com/yourusername/analizadorsentimientos/issues) or contact the author.

---

> © 2025 – Juan Camilo Cortés Sánchez

```

---

### ✅ ¿Qué debes cambiar?
- Reemplaza `https://github.com/yourusername/analizadorsentimientos.git` por la URL real de tu repositorio en GitHub.
- Asegúrate de subir también `requirements.txt`, `Visual.html` y el resto del código.

¿Te gustaría que también te ayude a generar el `requirements.txt` si aún no lo tienes?
```
