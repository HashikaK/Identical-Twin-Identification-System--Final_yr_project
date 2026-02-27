from flask import Flask, render_template

from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

from webcam import app as webcam_app
from img_upload import app as upload_app

# Create base app
main_app = Flask(__name__)

@main_app.route('/')
def home():
    return render_template('home.html')

# Combine apps
application = DispatcherMiddleware(main_app, {
    '/webcam': webcam_app,
    '/upload': upload_app
})

if __name__ == '__main__':
    run_simple('localhost', 5000, application, use_reloader=True, use_debugger=True)
