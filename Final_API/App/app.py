from app import create_app, socketio

app = create_app()

if __name__ == '__main__':
    socketio.run(app, host='192.168.31.53', port=5000)