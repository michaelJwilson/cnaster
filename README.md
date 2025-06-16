source .venv/bin/activate

uv pip install jupyter ipykernel

python -m ipykernel install --user --name cnaster --display-name "Python (cnaster)"

uv pip install --force-reinstall -e .