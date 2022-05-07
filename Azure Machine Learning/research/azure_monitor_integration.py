import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler

logger = logging.getLogger(__name__)

# TODO: replace the all-zero GUID with your instrumentation key.
logger.addHandler(AzureLogHandler(
    connection_string='InstrumentationKey=e17bd6de-8437-41d5-823b-6db68d4d87dc')
)

def valuePrompt():
    line = input("Error in Python script: #")
    logger.error(line)
    logger.critical(line)

def main():
    while True:
        valuePrompt()

if __name__ == "__main__":
    main()
