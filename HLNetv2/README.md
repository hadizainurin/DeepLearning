# Environment Preparation
Windows user (Windows 10/11 OS) is recommended.

Working environment:
1) Python 3.9
2) Visual Studio Code (VSC) 2022 ver.1.75.1

## 1.1 Create a new virtual Environment
- Open Visual Studio Code
    - Terminal > New Terminal
    - Create a new project folder (or select your own project folder), cd to the project folder in your terminal, and run the following command:
    ```
    # Before running the command
    mkdir HLNetv2 (No need this step)
    cd HLNetv2 (No need this step)
    # Run the command
    python3.9 -m venv HLNet_env
    ```
    - Delete your terminal. Now you should see (HLNet_env) in your terminal. If not restart VSC
    - Check if your environment is working
    ```
    pip list
    ```
    - If working you should see only necessary packages from Python 3.9

## 1.2 Install Torch
- Run the following commands
```
#upgrade pip to version 23.01
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```
- Run hlnetv2.py
