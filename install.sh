rm -rf build/
rm -rf dist/
python3 -m pip install -r requirements.txt
python3 -m setup build
python3 -m setup install
