from flask_cors import CORS
from flask import Flask, jsonify, request
from PeopleCounter import PeopleCounter
import threading
import logging
import time

app = Flask(__name__)
CORS(app)
counter = None
counter_thread = None
is_running = False

# Fixed config path
CONFIG_PATH = 'config.json'

# Queue for storing external events
external_events = []
external_events_lock = threading.Lock()


@app.route('/start', methods=['POST'])
def start_counter():
    """Start the people counting service"""
    global counter, counter_thread, is_running, external_events

    if is_running:
        return jsonify({'error': 'Counter is already running'}), 400

    # Get video source from request, with default
    data = request.get_json() or {}
    video_source = data.get('video_source', 'assets/samples/20231207153936_839_2.avi')

    try:
        # Clear external events
        with external_events_lock:
            external_events = []

        # Initialize the counter with fixed config
        counter = PeopleCounter(CONFIG_PATH)

        # Start counter in a separate thread
        counter_thread = threading.Thread(target=counter.start, args=(video_source,))
        counter_thread.daemon = True
        counter_thread.start()
        is_running = True

        # Start the external event processor
        event_thread = threading.Thread(target=process_external_events)
        event_thread.daemon = True
        event_thread.start()

        return jsonify({
            'message': 'Counter started successfully',
            'video_source': video_source
        }), 200
    except Exception as e:
        logging.error(f"Error starting counter: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/stop', methods=['POST'])
def stop_counter():
    """Stop the people counting service"""
    global counter, counter_thread, is_running

    if not is_running or counter is None:
        return jsonify({'error': 'Counter is not running'}), 400

    try:
        # Get results before stopping
        results = {
            'entries': counter.get_entry_count(),
            'exits': counter.get_exit_count()
        }

        # Stop the counter
        counter.stop()
        is_running = False

        # Wait for the thread to finish
        if counter_thread:
            counter_thread.join(timeout=5)  # Wait up to 5 seconds for the thread to finish

        counter = None
        counter_thread = None

        return jsonify({
            'message': 'Counter stopped successfully',
            'results': results
        }), 200
    except Exception as e:
        logging.error(f"Error stopping counter: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def get_status():
    """Get current status and counts"""
    global counter, is_running

    if not is_running or counter is None:
        return jsonify({
            'running': False,
            'entries': 0,
            'exits': 0
        }), 200

    return jsonify({
        'running': True,
        'entries': counter.get_entry_count(),
        'exits': counter.get_exit_count()
    }), 200


@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint"""
    return jsonify({'status': 'healthy'}), 200


@app.route('/api/people', methods=['POST'])
def handle_people_count():
    """Handle people count data in the original protocol format"""
    global counter, is_running, external_events

    try:
        data = request.get_json()

        # Validate required fields
        if not all(key in data for key in ['id', 'peopleState', 'objectType']):
            return jsonify({'error': 'Missing required fields'}), 400

        # Extract data
        camera_id = data['id']
        people_state = data['peopleState']  # 1 for entry, -1 for exit
        object_type = data['objectType']  # Should be 'person'

        # Validate object type
        if object_type != 'person':
            return jsonify({'error': 'Invalid object type'}), 400

        # Queue the event for processing
        with external_events_lock:
            external_events.append({
                'camera_id': camera_id,
                'people_state': people_state,
                'timestamp': time.time()
            })

        # Log the event
        logging.info(f"Received people count: Camera {camera_id}, State {people_state}, Type {object_type}")

        return jsonify({'status': 'success'}), 200

    except Exception as e:
        logging.error(f"Error handling people count: {str(e)}")
        return jsonify({'error': str(e)}), 500


def process_external_events():
    """Process external events in a separate thread"""
    global counter, is_running, external_events

    while is_running:
        # Process events if any are available
        events_to_process = []
        with external_events_lock:
            if external_events:
                events_to_process = external_events.copy()
                external_events = []

        # Process each event
        for event in events_to_process:
            try:
                if counter and is_running:
                    # Use the counter's logger to log the external event
                    if hasattr(counter, 'logger'):
                        if event['people_state'] == 1:
                            # Increment entry count in the counter's tracking
                            current_entry = counter.get_entry_count()
                            current_exit = counter.get_exit_count()
                            # Log entry (this will be as if the counter detected it)
                            counter.logger.log_entry(current_entry + 1, current_exit, event['timestamp'])
                            # Set the counter's internal count to match
                            counter.entry_count = current_entry + 1
                        elif event['people_state'] == -1:
                            # Increment exit count in the counter's tracking
                            current_entry = counter.get_entry_count()
                            current_exit = counter.get_exit_count()
                            # Log exit (this will be as if the counter detected it)
                            counter.logger.log_exit(current_entry, current_exit + 1, event['timestamp'])
                            # Set the counter's internal count to match
                            counter.exit_count = current_exit + 1
            except Exception as e:
                logging.error(f"Error processing external event: {str(e)}")

        # Sleep to avoid excessive CPU usage
        time.sleep(0.1)


def main():
    """Main entry point"""
    import argparse

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='People Counter Web Service')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the service on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the service on')
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()
