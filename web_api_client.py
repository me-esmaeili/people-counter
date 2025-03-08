import requests
from threading import Thread
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WebAPIClient:
    """Handles communication with web API"""
    def __init__(self, web_path, camera_id):
        self.web_path = web_path
        self.camera_id = camera_id
        self.sent_list = []

    def start(self):
        self.thread = Thread(target=self.run)
        self.thread.start()

    def run(self):
        while True:
            if self.sent_list:
                sent_one = self.sent_list.pop(0)
                max_retries = 3
                retry_delay = 2
                for attempt in range(max_retries):
                    try:
                        response = requests.post(
                            url=self.web_path,
                            json={"id": str(self.camera_id), "peopleState": sent_one, 'objectType': 'person'},
                            verify=False,
                            timeout=5
                        )
                        response.raise_for_status()
                        logging.info(f"Successfully sent event {sent_one} to {self.web_path}")
                        time.sleep(0.008)
                        break  # Exit retry loop on success
                    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                        # logging.error(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff: 2s, 4s, 8s
                        else:
                            # logging.warning(f"Max retries reached for event {sent_one}. Discarding.")
                            pass
                    except requests.exceptions.RequestException as e:
                        logging.error(f"Unexpected error: {e}")
                        time.sleep(2)
                        break
            else:
                time.sleep(0.0015)

    def add_event(self, event):
        self.sent_list.append(event)

if __name__ == "__main__":
    # Test the client
    client = WebAPIClient("http://example.com/api", "1")
    client.start()
    client.add_event(1)
    time.sleep(5)