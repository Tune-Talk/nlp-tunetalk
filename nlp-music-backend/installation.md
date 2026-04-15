<!-- Install libraries -->

python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows
pip install -r requirements.txt

<!-- Update requirements.txt once download library -->

pip freeze > requirements.txt

<!-- for run without run manual (nodeman in python) -->

flask --app test.py run --debug
