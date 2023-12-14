initialize_git:
	@echo "Initializing git..."
	git init 
	
install: 
	@echo "Installing..."
	poetry install
	poetry run pre-commit install
	poetry run pip install torch --index-url https://download.pytorch.org/whl/cu121

activate:
	@echo "Activating virtual environment"
	poetry shell

download_data:
	@echo "Downloading data..."
	wget https://maartensap.com/social-bias-frames/SBIC.v2.tgz -O data/raw/data.tgz
	tar -zxvf data/raw/data.tgz -C data/raw

setup: initialize_git install download_data

data/processed/xy.pkl: data/raw src/process.py
	@echo "Processing data..."
	python src/process.py

models/svc.pkl: data/processed/xy.pkl src/train_model.py
	@echo "Training model..."
	python src/train_model.py

notebooks/results.ipynb: models/svc.pkl src/run_notebook.py
	@echo "Running notebook..."
	python src/run_notebook.py

pipeline: data/processed/xy.pkl models/svc.pkl notebooks/results.ipynb

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache