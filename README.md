# law-agent
pip install -r requirements.txt
venv\Scripts\activate
python -m uvicorn main:app --reload
python -m uvicorn src.main:app --reload

 http://127.0.0.1:8000/query

 {
  "question": "What is the significance of the Sangam age in Tamil history?"
}