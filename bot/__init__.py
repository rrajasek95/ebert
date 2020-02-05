from flask import Flask

from .bot import bp
def create_app():
    app = Flask(__name__)

    app.register_blueprint(bot.bp)
    return app