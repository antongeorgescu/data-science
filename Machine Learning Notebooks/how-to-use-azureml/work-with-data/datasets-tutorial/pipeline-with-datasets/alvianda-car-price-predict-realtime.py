import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
data = {
    "Inputs": {
        "WebServiceInput0":
        [
            {
                "symboling": "3",
                "normalized-losses": "1.0",
                "make": "alfa-romero",
                "fuel-type": "gas",
                "aspiration": "std",
                "num-of-doors": "two",
                "body-style": "convertible",
                "drive-wheels": "rwd",
                "engine-location": "front",
                "wheel-base": "88.6",
                "length": "168.8",
                "width": "64.1",
                "height": "48.8",
                "curb-weight": "2548",
                "engine-type": "dohc",
                "num-of-cylinders": "four",
                "engine-size": "130",
                "fuel-system": "mpfi",
                "bore": "3.47",
                "stroke": "2.68",
                "compression-ratio": "9.0",
                "horsepower": "111.0",
                "peak-rpm": "5000.0",
                "city-mpg": "21",
                "highway-mpg": "27",
                "price": "13495.0"
            },
        ]
    },
    "GlobalParameters": {
        "split_ratio_2": "0.7"
    }
}

body = str.encode(json.dumps(data))

url = 'http://134db94f-5709-44d3-8c9e-812eae7352dc.canadacentral.azurecontainer.io/score'
api_key = 'Z9F8OBT7L51nxccDadvqARnrd8Z2Mnl2' # Replace this with the API key for the web service
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
    print(error.read().decode("utf8", 'ignore'))
