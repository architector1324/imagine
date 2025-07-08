#!/bin/bash

mkdir -p build

python -m venv ./build/venv
source ./build/venv/bin/activate

pip install --upgrade pip
pip install pyinstaller
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install diffusers transformers Pillow requests ollama

cp ./src/imagine.py ./src/imagine_server.py ./src/imagine_run.py ./src/imagine_enhance.py ./src/imagine_server_defs.py ./src/imagine_list.py -t build/

cd build
pyinstaller --onefile --collect-all diffusers --collect-all ollama --hidden-import imagine_run --hidden-import imagine_server --hidden-import imagine_server_defs --hidden-import imagine_enhance --hidden-import imagine_list --add-data="imagine_run.py:." --add-data="imagine_list.py:." --add-data="imagine_server.py:." --add-data="imagine_server_defs.py:." --add-data="imagine_enhance.py:." imagine.py

mv dist/imagine ../imagine
