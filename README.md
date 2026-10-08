# AI Desire Detector — Deep Learning Project

## What it does
This beginner-friendly project uses a small neural network to classify the **goal or preference expressed in a sentence** into one of four demo categories:
- Learning & Career
- Travel & Adventure
- Food & Cooking
- Fitness & Health

It does **not** read minds or prove what a person secretly wants. It classifies wording based on the labelled examples in the dataset.

## Requirements
- Python 3.10 or 3.11 recommended
- Internet connection for installing packages
- About 2–4 GB of free disk space may be needed for TensorFlow and dependencies

## Run on Windows
1. Extract the ZIP.
2. Open Command Prompt in the extracted folder.
3. Create a virtual environment:
   `python -m venv .venv`
4. Activate it:
   `.venv\\Scripts\\activate`
5. Install packages:
   `pip install -r requirements.txt`
6. Train the neural network:
   `python train_model.py`
7. Start the app:
   `streamlit run app.py`
8. A browser tab should open. Enter a sentence and click **Detect expressed preference**.

## Run on macOS/Linux
Use:
- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `python train_model.py`
- `streamlit run app.py`

## Project files
- `app.py` — interactive Streamlit interface
- `train_model.py` — text preprocessing and neural-network training
- `data/training_data.csv` — small educational example dataset
- `project_report.md` — report template/content
- `viva_questions.md` — viva preparation

## Important academic note
The included dataset is small and illustrative. Results are for demonstration, not real psychological inference. For a stronger project, collect a larger, consent-based dataset, remove personal identifiers, split by person to avoid leakage, and evaluate precision/recall and a confusion matrix.
