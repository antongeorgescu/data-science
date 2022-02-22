import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

# allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
data = {
    "Inputs": {
        "data":
        [
            {
                "Date": "2021-01-05T00:00:00.000Z",
                "Open": "255.5"
            },
        ]
    },
    "GlobalParameters": {
        "quantiles": "0.025,0.975"
    }
}

body = str.encode(json.dumps(data))

url = 'http://9e678cf9-440f-49e9-b031-39af1cbf1ffa.eastus.azurecontainer.io/score'
api_key = 'WwfuMUTM612QxVRneYizn9kLnsD4vhzb' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    # print(json.loads(error.read().decode("utf8", 'ignore')))
