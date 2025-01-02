# Style Transfer Web Application

This is a simple web application built using Flask to demonstrate the style transfer process between two images. Users can upload a content image and a style image, and the model will apply the style to the content image.

## Features

- **Style Transfer:** Apply style transfer using deep learning models.
- **Upload Interface:** Users can upload images directly from the web interface.
- **Stylized Image Display:** View the result after style transfer.
- **Download Result:** Option to download the stylized image.

## Requirements

This project requires Python 3.7+ and the following Python libraries:

- Flask
- TensorFlow
- Pillow
- werkzeug

You can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt

Installation

Follow the steps below to set up the project locally.
1. Clone the Repository

Clone this repository to your local machine:

git clone https://github.com/yourusername/style-transfer-project.git

2. Navigate to the Project Directory

cd style-transfer-project

3. Install Dependencies

Install all required dependencies:

pip install -r requirements.txt

4. Run the Flask Application

Start the Flask application by running:

python app.py

5. Open the Application

After running the Flask application, open your browser and visit:

http://127.0.0.1:5000/

Deployment

For deployment, the application can be hosted on platforms like Heroku or PythonAnywhere.
Heroku Deployment

To deploy your application to Heroku, follow these steps:

    Create a New Heroku Project
    Visit Heroku and create a new project or log in to your existing account.

    Install the Heroku CLI
    If you don't have the Heroku CLI installed, you can download and install it from here.

    Initialize the Git Repository

    If you haven't already, initialize a Git repository in your project folder:

git init
git add .
git commit -m "Initial commit"

Login to Heroku CLI

Log in to Heroku via the CLI:

heroku login

Deploy to Heroku

Create a Heroku application and push your code:

heroku create your-app-name
git push heroku master

Open the Application

After deployment, you can open your application by running:

    heroku open

License

This project is licensed under the MIT License.
