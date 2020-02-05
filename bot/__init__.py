from flask import Flask
from flask_executor import Executor

from .bot import bp
def create_app():
    app = Flask(__name__)

    app.register_blueprint(bot.bp)

    app.config['EXECUTOR'] = Executor(app)
    app.config['EXECUTOR_PROPAGATE_EXCEPTIONS'] = True 
    return app