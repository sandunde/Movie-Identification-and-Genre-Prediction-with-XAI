Application Documentation

Overview

This document provides detailed information about the setup, dependencies, and usage of the application. The application combines a ReactJS frontend with a Flask backend to perform various natural language processing (NLP) and machine learning tasks.

## Installation

### Backend (Flask)

1. Ensure you have Python installed. If not, download and install it from [here](https://www.python.org/downloads/).
2. Clone the repository to your local machine.
3. Navigate to the backend directory:

    ```bash
    cd flask-server
    ```

4. Create a virtual environment:

    ```bash
    python -m venv venv
    ```

5. Activate the virtual environment:

    - On Windows:

    ```bash
    venv\Scripts\activate
    ```

    - On macOS and Linux:

    ```bash
    source venv/bin/activate
    ```

6. Install the required Python packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

### Frontend (ReactJS)

1. Ensure you have Node.js and npm installed. If not, download and install them from [here](https://nodejs.org/).
2. Navigate to the frontend directory:

    ```bash
    cd client
    ```

3. Install the required npm packages:

    ```bash
    npm install
    ```

## Usage

1. Start the Flask backend server:

    ```bash
    cd flask-server
    python app.py
    ```

2. Start the ReactJS frontend server:

    ```bash
    cd client
    npm start
    ```

3. Access the application by visiting http://localhost:3000 in your web browser.

## Libraries and Frameworks

### Backend (Flask)

- Flask
- pandas
- scikit-learn
- keras
- seaborn
- matplotlib
- nltk
- transformers
- torch
- Flask-CORS
- captum
- torchaudio
- python-Levenshtein

### Frontend (ReactJS)

- React
- axios

## Contributors

- Sandun De Silva

## License

This project is licensed under the [MIT License](LICENSE).