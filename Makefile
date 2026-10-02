install:
	python -m pip install -r requirements.txt

test:
	pytest -q

dashboard:
	streamlit run dashboard.py

api:
	uvicorn api.main:app --reload

lint:
	python -m compileall src api dashboard.py
