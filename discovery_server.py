import socket
import threading
import time
import pickle

# --- Configuration ---
HOST = '0.0.0.0'       # Listen on all interfaces
PORT = 5000            # Discovery server port
DISCOVERY_TIME = 20    # Seconds to collect clients before starting

clients = []           # List of dicts: {"ip": ..., "port": ..., "rank": ..., "conn": ...}
lock = threading.Lock()

def handle_client(conn, addr):
    """
    Handle a client connection: receive its info and keep the connection open
    until the discovery period ends, then send a start signal.
    """
    try:
        # Receive client info (pickled dict: {"ip":..., "port":..., "rank":...})
        data = conn.recv(1024)
        client_info = pickle.loads(data)
        client_info["conn"] = conn  # Keep the socket for signaling

        with lock:
            # Avoid duplicates (by rank)
            if all(c["rank"] != client_info["rank"] for c in clients):
                clients.append(client_info)
        
        print(f"[Discovery] Registered {client_info}, total clients: {len(clients)}")
    except Exception as e:
        print(f"[Discovery] Error handling {addr}: {e}")

def main():
    print(f"[Discovery] Server starting on {HOST}:{PORT}...")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen()

    print(f"[Discovery] Listening for clients for {DISCOVERY_TIME} seconds...")
    server.settimeout(DISCOVERY_TIME)

    try:
        while True:
            try:
                conn, addr = server.accept()
            except socket.timeout:
                print("[Discovery] Discovery time ended!")
                break
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
    finally:
        server.close()

    # --- Signal all clients to start ---
    print("[Discovery] Sending start signal to all clients...")
    with lock:
        for c in clients:
            try:
                # Remove the "conn" field before sending
                info_to_send = [ {k:v for k,v in client.items() if k != "conn"} for client in clients ]
                c["conn"].sendall(pickle.dumps(info_to_send))
                c["conn"].close()
            except Exception as e:
                print(f"[Discovery] Error signaling client {c['rank']}: {e}")

    print("[Discovery] Final registered clients:")
    for c in clients:
        print(c)

if __name__ == "__main__":
    main()
