#!/bin/bash

mkdir -p build

python -m venv ./build/venv
source ./build/venv/bin/activate

pip install --upgrade pip
pip install pyinstaller
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install diffusers transformers Pillow requests ollama

cp ./src/imagine.py ./src/imagine_server.py ./src/imagine_run.py ./src/imagine_enhance.py -t build/

cd build
pyinstaller --clean --onefile --collect-all diffusers --collect-all ollama --hidden-import imagine_run --hidden-import imagine_server --hidden-import imagine_enhance --add-data="imagine_run.py:." --add-data="imagine_server.py:." --add-data="imagine_enhance.py:." imagine.py

cp dist/imagine ../imagine
