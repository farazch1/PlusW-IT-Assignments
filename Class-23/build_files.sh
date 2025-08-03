pip install --upgrade pip
python -m venv myvenv
source venv/bin/activate
pip install -r requirements.txt
python manage.py collectstatic