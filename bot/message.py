import os
import requests
import json

def call_send_api(request, app):
    url = "https://graph.facebook.com/v6.0/me/messages?access_token=%s" % os.getenv('PAGE_ACCESS_TOKEN', '')
    app.logger.info("POST to URL : %s" % url)
    app.logger.info("Payload: %s" % json.dumps(request))

    r = requests.post(url, json=request)
    app.logger.info(r.status_code)
    app.logger.info(r.text)