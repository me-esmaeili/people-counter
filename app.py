# webservice.py
from flask_cors import CORS
from flask import Flask, jsonify, request
from PeopleCounter import PeopleCounter
import threading

app = Flask(__name__)
CORS(app)
counter = None
counter_thread = None
is_running = False

# Fixed config path
CONFIG_PATH = 'config.json'


@app.route('/start', methods=['POST'])
def start_counter():
    """Start the people counting service"""
    global counter, counter_thread, is_running

    if is_running:
        return jsonify({'error': 'Counter is already running'}), 400

    # Get video source from request, with default
    data = request.get_json() or {}
    video_source = data.get('video_source', 'assets/samples/20231207153936_839_2.avi')

    try:
        # Initialize the counter with fixed config
        counter = PeopleCounter(CONFIG_PATH)

        # Start counter in a separate thread
        counter_thread = threading.Thread(target=counter.start, args=(video_source,))
        counter_thread.daemon = True
        counter_thread.start()
        is_running = True

        return jsonify({
            'message': 'Counter started successfully',
            'video_source': video_source
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/stop', methods=['POST'])
def stop_counter():
    """Stop the people counting service"""
    global counter, counter_thread, is_running

    if not is_running or counter is None:
        return jsonify({'error': 'Counter is not running'}), 400

    try:
        # Stop the counter and get results
        results = counter.stop()
        is_running = False

        # Wait for the thread to finish
        if counter_thread:
            counter_thread.join()

        counter = None
        counter_thread = None

        return jsonify({
            'message': 'Counter stopped successfully',
            'results': results
        }), 200
    except Exception as e:
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


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='People Counter Web Service')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the service on')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the service on')
    args = parser.parse_args()

    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()