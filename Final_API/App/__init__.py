
# //__init__.py
from flask import Flask
from flask_socketio import SocketIO

socketio = SocketIO( cors_allowed_origins="*")

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your_secret_key'
    
    # Initialize SocketIO
    socketio.init_app(app)

    # Register blueprints
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app