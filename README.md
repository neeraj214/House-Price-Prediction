# House Price Prediction

Production-ready Streamlit application to predict house prices from user inputs or batch CSV files.

## Features

- Streamlit UI with manual input and batch CSV prediction tabs
- Indian currency formatting
- Works with a trained pipeline (`models/best_pipeline.pkl`) or saved model + columns (`models/model_<stem>.pkl` and `models/feature_columns_<stem>.json`)

## Project Structure

- `app.py`: Streamlit app entrypoint
- `models/`: Trained artifacts (not included)
- `src/`: Training and evaluation utilities
- `backend/`, `frontend/`: Optional FastAPI/Next.js prototype (not required for Streamlit deployment)

## Local Setup

```bash
python -m pip install -r requirements.txt
python app.py
```

This launches the Streamlit server using the built-in launcher.

### Model Artifacts

- Manual Input tab looks for `models/best_pipeline.pkl`
- Batch Upload tab expects:
  - `models/model_<stem>.pkl`
  - `models/feature_columns_<stem>.json`

## Deployment

### Streamlit Community Cloud
1. Push this repo to GitHub.
2. Create a new app at https://share.streamlit.io/ and connect your repo.
3. App file path: `app.py`
4. Python version: 3.10
5. Deploy.

### Hugging Face Spaces
1. Create a new Space (type: Streamlit).
2. Select your GitHub repo as the source, or upload files.
3. Ensure `requirements.txt` and `app.py` are present.
4. The app will build and start automatically.

### Render
1. Create a new Web Service from your GitHub repo.
2. Environment: Python
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Deploy.

## GitHub

Initialize and push (replace ORIGIN_URL):
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin ORIGIN_URL
git push -u origin main
```

## Troubleshooting

- No model found: upload `models/best_pipeline.pkl` for manual input, or a pair of `model_<stem>.pkl` and `feature_columns_<stem>.json` for batch mode.
- Port binding on Render: ensure the start command uses `$PORT` and `0.0.0.0`.
