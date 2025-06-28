#!/bin/bash

mkdir -p build

python -m venv ./build/venv
source ./build/venv/bin/activate

pip install --upgrade pip
pip install pyinstaller
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install diffusers transformers Pillow requests

cp ./src/imagine.py ./src/imagine_server.py ./src/imagine_run.py -t build/

cd build
pyinstaller --clean --onefile --collect-all diffusers --hidden-import imagine_run --hidden-import imagine_server --add-data="imagine_run.py:." --add-data="imagine_server.py:." imagine.py

cp dist/imagine ../imagine
