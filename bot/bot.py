from flask import (
    Blueprint
)

bp = Blueprint('bot', __name__)

@bp.route('/')
def hello_world():
    return 'Hello, World!'