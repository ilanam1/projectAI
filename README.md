סדר הרצה :

cd backend

rmdir /s /q venv

python -m venv venv

venv\Scripts\activate.bat

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

python train_local_model.py

uvicorn main:app --reload --host 0.0.0.0 --port 8000




in other terminal:

cd frontend

npm run dev





# Fake Review Detector

A full-stack web application for detecting fake reviews using Machine Learning. Built with FastAPI (Python) backend and React + Tailwind CSS frontend.

## Project Structure

```
binaproj/
├── backend/              # FastAPI backend
│   ├── main.py          # FastAPI application with /predict endpoint
│   ├── ml_model.py      # ML model placeholder (ready for Hugging Face integration)
│   ├── requirements.txt # Python dependencies
│   └── .gitignore
├── frontend/            # React + Vite frontend
│   ├── src/
│   │   ├── components/
│   │   │   └── ReviewAnalyzer.jsx  # Main UI component
│   │   ├── App.jsx      # Root component
│   │   ├── main.jsx     # Entry point
│   │   └── index.css    # Tailwind CSS imports
│   ├── package.json     # Node.js dependencies
│   ├── vite.config.js   # Vite configuration
│   ├── tailwind.config.js
│   └── postcss.config.js
└── README.md
```

## Features

- **Backend API**: RESTful API with `/predict` endpoint
- **ML Model Ready**: Placeholder function ready for Hugging Face model integration
- **Modern UI**: Clean, responsive interface with Tailwind CSS
- **Real-time Analysis**: Instant review analysis with loading states
- **Visual Feedback**: Color-coded results (Green for Real, Red for Fake) with confidence bars

## Prerequisites

- **Python 3.8+** (for backend)
- **Node.js 16+** and **npm** (for frontend)

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the FastAPI server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

   The API will be available at `http://localhost:8000`
   - API Documentation: `http://localhost:8000/docs` (Swagger UI)
   - Alternative Docs: `http://localhost:8000/redoc`

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:5173`

## Running Both Servers

### Option 1: Separate Terminals (Recommended)

1. **Terminal 1** - Backend:
   ```bash
   cd backend
   # Activate virtual environment if using one
   uvicorn main:app --reload --port 8000
   ```

2. **Terminal 2** - Frontend:
   ```bash
   cd frontend
   npm run dev
   ```

### Option 2: Using npm scripts (if you prefer)

You can create a script to run both servers. However, for development, separate terminals are recommended for better log visibility.

## Usage

1. Start both backend and frontend servers (see above)
2. Open your browser to `http://localhost:5173`
3. Enter a review text in the textarea
4. Click "Analyze Review"
5. View the results:
   - **Green badge**: Real Review
   - **Red badge**: Fake Review
   - Confidence percentage bar
   - Detailed probability breakdown

## API Endpoints

### `POST /predict`

Analyze a review text to determine if it's fake or real.

**Request Body:**
```json
{
  "text": "This product is amazing! I love it so much."
}
```

**Response:**
```json
{
  "is_fake": false,
  "confidence": 0.85,
  "probability": 0.15
}
```

**Fields:**
- `is_fake`: Boolean indicating if the review is detected as fake
- `confidence`: Confidence score (0.0 to 1.0) - distance from decision threshold
- `probability`: Probability that the review is fake (0.0 to 1.0)

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

## Integrating Your ML Model

📖 **למדריך מפורט בעברית, ראה:** [`backend/MODEL_SETUP.md`](backend/MODEL_SETUP.md)

The backend is structured to easily integrate a Hugging Face model. To add your model:

1. **Install ML dependencies** (uncomment in `requirements.txt`):
   ```txt
   torch==2.1.0
   transformers==4.35.0
   numpy==1.24.3
   ```

2. **Update `backend/ml_model.py`**:

   Replace the placeholder function with your model:

   ```python
   from transformers import pipeline
   import torch

   # Load your model (do this once, outside the function)
   # You can load it at module level or use a singleton pattern
   classifier = None

   def load_model():
       global classifier
       if classifier is None:
           classifier = pipeline(
               "text-classification",
               model="your-model-name",  # e.g., "distilbert-base-uncased-finetuned-sst-2-english"
               device=0 if torch.cuda.is_available() else -1
           )
       return classifier

   def detect_fake_review(text: str) -> float:
       """
       Detect if a review is fake using a trained ML model.
       """
       model = load_model()
       result = model(text)
       
       # Adjust based on your model's output format
       # If your model returns labels like "FAKE"/"REAL" or "LABEL_0"/"LABEL_1"
       if result[0]['label'] == 'FAKE' or result[0]['label'] == 'LABEL_1':
           return result[0]['score']
       else:
           return 1 - result[0]['score']
   ```

3. **For PyTorch models**, you can also load a custom model:

   ```python
   from transformers import AutoTokenizer, AutoModelForSequenceClassification
   import torch

   tokenizer = AutoTokenizer.from_pretrained("your-model-path")
   model = AutoModelForSequenceClassification.from_pretrained("your-model-path")
   
   def detect_fake_review(text: str) -> float:
       inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
       with torch.no_grad():
           outputs = model(**inputs)
           probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
           # Assuming label 1 is "fake"
           return probabilities[0][1].item()
   ```

## Development

### Backend Development

- The FastAPI server runs with auto-reload enabled (`--reload` flag)
- API documentation is available at `/docs` (Swagger UI)
- CORS is configured to allow requests from `localhost:5173` and `localhost:3000`

### Frontend Development

- Vite provides hot module replacement (HMR) for instant updates
- Tailwind CSS is configured and ready to use
- The API URL is set to `http://localhost:8000/predict` in `ReviewAnalyzer.jsx`

## Production Build

### Frontend

```bash
cd frontend
npm run build
```

The built files will be in the `frontend/dist` directory.

### Backend

For production, use a production ASGI server like Gunicorn with Uvicorn workers:

```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Troubleshooting

### CORS Errors

If you encounter CORS errors, ensure:
- Backend is running on port 8000
- Frontend is running on port 5173 (or update CORS origins in `backend/main.py`)

### Port Already in Use

- Backend: Change port with `--port 8001` (and update frontend API URL)
- Frontend: Vite will automatically suggest an alternative port

### Module Not Found Errors

- Backend: Ensure virtual environment is activated and dependencies are installed
- Frontend: Run `npm install` in the frontend directory

## License

This project is part of a group project for educational purposes.

## Contributing

This is a group project. Coordinate with your team members before making major changes.

