import os

from flask import (
    Blueprint, request, jsonify
)

bp = Blueprint('bot', __name__)

@bp.route('/')
def hello_world():
    return 'Hello, World!'

def handle_verification():
    VERIFY_TOKEN = os.getenv('VERIFY_TOKEN')
    mode = request.args.get('hub.mode', '')
    token = request.args.get('hub.verify_token', '')
    challenge = request.args.get('hub.challenge', '')

    if mode and token:
        if mode == 'subscribe' and token == VERIFY_TOKEN:
            return challenge, 200
        else:
            return 'Forbidden', 403
    else:
        return 'Forbidden', 403


def handle_message():
    content = request.json
    
    if content['object'] == "page":
        for entry in content['entry']:
            event = entry['messaging'][0]
            print(event)
        return 'EVENT_RECEIVED'
    else:
        return 'Page Not Found', 404

@bp.route('/webhook', methods=['POST', 'GET'])
def handle_webhook():
    if request.method == "GET":
        return handle_verification()
    else:
        return handle_message()