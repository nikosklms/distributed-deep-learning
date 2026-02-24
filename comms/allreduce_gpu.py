# Clean, callback-based fault-tolerant ring allreduce

import socket
import threading
import time
import numpy as np
import pickle
import cupy as cp
import os
import queue
import struct
import uuid
import random
from pathlib import Path
from enum import Enum

# Import fault tolerance modules
from comms.udp_reliable import ReliableUDP
from comms.gosip import GossipProtocol
from comms.heartbeats import HeartbeatMonitor

# Fast sockets (C extension for faster I/O with GIL release)
try:
    import comms.fast_net as fast_net
    HAS_FAST_NET = True
except ImportError:
    HAS_FAST_NET = False
    print("WARNING: fast_net C extension not found, using pure Python sockets")

# Configuration
DISCOVERY_IP = os.getenv("DISCOVERY_IP", "127.0.0.1")
DISCOVERY_PORT = 5000
BASE_PORT = 6000
UDP_PORT_OFFSET = 2000
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)
SOCKET_RECV_TIMEOUT = 10.0

# Framing constants
MSG_TYPE_DATA = 1
FRAME_HEADER_FMT = ">BII"
FRAME_HEADER_SIZE = struct.calcsize(FRAME_HEADER_FMT)


class State(Enum):
    """Simple 2-state machine"""
    NORMAL = 1
    RECOVERING = 2


def send_chunk_framed(sock, chunk_list, iteration_id=0, chunk_id=0):
    """Send a chunk of arrays over TCP with framing"""
    shapes = []
    dtypes = []
    total_bytes = 0
    prepared = []
    
    for arr in chunk_list:
        a = np.array(arr, copy=False)
        if a.dtype != np.float16:
            a = a.astype(np.float16)
        if not a.flags['C_CONTIGUOUS']:
            a = np.ascontiguousarray(a)
        prepared.append(a)
        shapes.append(a.shape)
        dtypes.append(str(a.dtype))
        total_bytes += a.size * a.dtype.itemsize

    meta = {'shapes': shapes, 'dtypes': dtypes, 'iter': iteration_id, 'chunk': chunk_id}
    meta_bytes = pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL)

    header = struct.pack(FRAME_HEADER_FMT, MSG_TYPE_DATA, len(meta_bytes), total_bytes)
    sock.sendall(header)
    sock.sendall(meta_bytes)
    for a in prepared:
        sock.sendall(a.ravel().tobytes())


def send_chunk_fast(sock, chunk_list):
    """Send using C extension (faster, releases GIL)"""
    ready_list = []
    for arr in chunk_list:
        a = np.array(arr, copy=False)
        if a.dtype != np.float16:
            a = a.astype(np.float16)
        if not a.flags['C_CONTIGUOUS']:
            a = np.ascontiguousarray(a)
        ready_list.append(a.ravel())
    fast_net.send_list(sock.fileno(), ready_list)


def recv_and_unpack_fast(sock, template_chunk_list, recv_buffer):
    """Receive using C extension (faster, releases GIL).
    
    Args:
        sock: Socket to receive from
        template_chunk_list: List of templates for shape info
        recv_buffer: Preallocated numpy float16 buffer (from self.fast_recv_buffer)
    """
    total_elements = sum(arr.size for arr in template_chunk_list)
    
    # Use a view into the preallocated buffer
    buffer_view = recv_buffer[:total_elements]
    
    # Set timeout - C code now properly handles EAGAIN/EWOULDBLOCK
    old_timeout = sock.gettimeout()
    try:
        sock.settimeout(SOCKET_RECV_TIMEOUT)
        success = fast_net.recv_into_array(sock.fileno(), buffer_view)
    finally:
        sock.settimeout(old_timeout)
    
    if not success:
        raise EOFError("Connection broken during fast recv")
    
    restored = []
    offset = 0
    for template in template_chunk_list:
        size = template.size
        arr = buffer_view[offset:offset+size].reshape(template.shape).copy()
        restored.append(arr)
        offset += size
    return None, restored  # Return (meta, data) to match framed API


def recv_exact_into(sock, target_mv, n, timeout=None):
    """Receive exactly n bytes into target memoryview"""
    old_timeout = sock.gettimeout()
    try:
        if timeout is not None:
            sock.settimeout(timeout)
        mv = target_mv
        total = 0
        while total < n:
            got = sock.recv_into(mv[total:n], n - total)
            if got == 0:
                raise EOFError("Socket closed while reading")
            total += got
    except socket.timeout:
        raise TimeoutError("recv_exact_into timeout")
    finally:
        try:
            sock.settimeout(old_timeout)
        except OSError:
            pass


def recv_and_unpack_framed(sock, template_chunk_list, recv_buffer_bytearray, timeout=None):
    """Receive and unpack a framed message"""
    old_timeout = sock.gettimeout()
    try:
        if timeout is not None:
            sock.settimeout(timeout)

        header = b''
        while len(header) < FRAME_HEADER_SIZE:
            chunk = sock.recv(FRAME_HEADER_SIZE - len(header))
            if not chunk:
                raise EOFError("Socket closed while reading header")
            header += chunk
        
        msg_type, meta_len, payload_len = struct.unpack(FRAME_HEADER_FMT, header)
        if msg_type != MSG_TYPE_DATA:
            raise ValueError(f"Unexpected message type: {msg_type}")

        meta_bytes = b''
        while len(meta_bytes) < meta_len:
            chunk = sock.recv(meta_len - len(meta_bytes))
            if not chunk:
                raise EOFError("Socket closed while reading meta")
            meta_bytes += chunk
        meta = pickle.loads(meta_bytes)

        if payload_len > len(recv_buffer_bytearray):
            raise ValueError(f"Payload too large: {payload_len} bytes")
        
        mv = memoryview(recv_buffer_bytearray)
        recv_exact_into(sock, mv, payload_len, timeout=timeout)

        shapes = meta.get('shapes', [])
        dtypes = meta.get('dtypes', [])
        restored = []
        offset_bytes = 0
        
        for shp, dt in zip(shapes, dtypes):
            dtype = np.dtype(dt)
            elem_count = int(np.prod(shp))
            byte_count = int(elem_count * dtype.itemsize)
            slice_mv = mv[offset_bytes: offset_bytes + byte_count]
            arr = np.frombuffer(slice_mv, dtype=dtype, count=elem_count)
            arr = arr.reshape(shp).copy()
            restored.append(arr)
            offset_bytes += byte_count

        return meta, restored

    except socket.timeout:
        raise TimeoutError("Receive timeout")
    finally:
        try:
            sock.settimeout(old_timeout)
        except OSError:
            pass


def chunk_list(lst, n):
    """Split list into n chunks"""
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


def recv_all_pickle(sock):
    """Receive a pickled object with length prefix"""
    raw_len = b''
    while len(raw_len) < 4:
        chunk = sock.recv(4 - len(raw_len))
        if not chunk:
            return None
        raw_len += chunk
    msg_len = struct.unpack('>I', raw_len)[0]
    data = b''
    while len(data) < msg_len:
        chunk = sock.recv(msg_len - len(data))
        if not chunk:
            return None
        data += chunk
    return pickle.loads(data)


class CheckpointManager:
    """Manages checkpoint saving/loading with 2-file rotation.
    
    Keeps the 2 most recent checkpoints to handle the case where a node
    crashes before checkpointing while others have already checkpointed.
    
    File naming: checkpoint_rank_{rank}_iter_{iteration}.pkl
    """
    
    def __init__(self, rank, keep_count=2):
        self.rank = rank
        self.keep_count = keep_count  # Number of checkpoints to keep
        self.lock = threading.Lock()
    
    def _get_pattern(self):
        """Get glob pattern for finding checkpoints for this rank"""
        return f"checkpoint_rank_{self.rank}_iter_*.pkl"
    
    def _get_checkpoint_path(self, iteration):
        """Generate checkpoint path for specific iteration"""
        return CHECKPOINT_DIR / f"checkpoint_rank_{self.rank}_iter_{iteration}.pkl"
    
    def _extract_iteration(self, path):
        """Extract iteration number from checkpoint filename"""
        # Format: checkpoint_rank_N_iter_M.pkl
        try:
            name = path.stem  # checkpoint_rank_N_iter_M
            parts = name.split('_iter_')
            if len(parts) == 2:
                return int(parts[1])
        except (ValueError, IndexError):
            pass
        return -1
    
    def _get_all_checkpoints(self):
        """Get all checkpoint files for this rank, sorted by iteration (newest first)"""
        pattern = self._get_pattern()
        files = list(CHECKPOINT_DIR.glob(pattern))
        # Sort by iteration number, descending (newest first)
        files.sort(key=lambda p: self._extract_iteration(p), reverse=True)
        return files
    
    def _cleanup_old_checkpoints(self):
        """Delete checkpoints beyond keep_count (keep only newest)"""
        files = self._get_all_checkpoints()
        # Keep the first `keep_count` files, delete the rest
        for old_file in files[self.keep_count:]:
            try:
                old_file.unlink()
            except OSError:
                pass  # Ignore deletion errors

    def exists(self):
        """Check if any checkpoint exists for this rank"""
        with self.lock:
            files = self._get_all_checkpoints()
            return len(files) > 0

    def save(self, iteration, model_state, rng_states=None):
        """Save checkpoint atomically with iteration-based naming"""
        with self.lock:
            checkpoint = {
                'rank': self.rank,
                'iteration': iteration,
                'model_state': model_state,
                'rng_states': rng_states or {},
                'timestamp': time.time()
            }
            
            checkpoint_path = self._get_checkpoint_path(iteration)
            temp_file = checkpoint_path.with_suffix('.tmp')
            
            with open(temp_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            temp_file.replace(checkpoint_path)
            
            # Cleanup old checkpoints (keep only newest keep_count)
            self._cleanup_old_checkpoints()
            
        return time.time() - checkpoint['timestamp']

    def load_latest(self):
        """Load the most recent checkpoint"""
        with self.lock:
            files = self._get_all_checkpoints()
            if not files:
                return None
            # First file is the newest (sorted descending by iteration)
            with open(files[0], 'rb') as f:
                return pickle.load(f)
    
    def load(self, iteration=None):
        """Load checkpoint - latest if no iteration specified, or specific iteration"""
        if iteration is None:
            return self.load_latest()
        
        with self.lock:
            checkpoint_path = self._get_checkpoint_path(iteration)
            if not checkpoint_path.exists():
                return None
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
    
    def get_available_iterations(self):
        """Get list of available checkpoint iterations (newest first)"""
        with self.lock:
            files = self._get_all_checkpoints()
            return [self._extract_iteration(f) for f in files]

    def delete_all(self):
        """Delete ALL checkpoints for this rank"""
        with self.lock:
            for f in self._get_all_checkpoints():
                try:
                    f.unlink()
                except OSError:
                    pass
    
    def delete(self):
        """Alias for delete_all() for backward compatibility"""
        self.delete_all()


class FaultTolerantRingAllReducer:
    """
    Callback-based fault-tolerant ring allreduce
    
    Key features:
    - Automatic checkpointing via get_model_state_fn callback
    - Automatic rollback via set_model_state_fn callback
    - Simple 2-state machine (NORMAL, RECOVERING)
    - No thread barriers, just queues and events
    - Application never sees failures or recovery
    """
    
    def __init__(self, rank, world_size, get_model_state_fn, set_model_state_fn, 
                 checkpoint_interval=200, verbose=True):
        """
        Initialize fault-tolerant communicator
        
        Args:
            rank: This node's rank
            world_size: Total number of nodes
            get_model_state_fn: Callback to get model state dict for checkpointing
            set_model_state_fn: Callback to restore model state dict during recovery
            checkpoint_interval: Save checkpoint every N iterations
            verbose: Enable debug logging
        """
        self.rank = rank
        self.world_size = world_size
        self.get_model_state_fn = get_model_state_fn
        self.set_model_state_fn = set_model_state_fn
        self.checkpoint_interval = checkpoint_interval
        self.verbose = verbose
        
        self.run_id = str(uuid.uuid4())
        
        # Simple state machine
        self.state = State.NORMAL
        self.state_lock = threading.Lock()
        
        # Single iteration counter
        self.iteration = 0
        self.iteration_lock = threading.Lock()
        
        # Checkpoint manager
        self.checkpoint_mgr = CheckpointManager(rank)
        self.is_recovering = self.checkpoint_mgr.exists()
        
        # Fault tolerance components
        self.udp_port = BASE_PORT + UDP_PORT_OFFSET + self.rank
        self.udp = ReliableUDP(self.rank, self.udp_port, verbose=False)
        self.gossip = GossipProtocol(self.rank, self.world_size, self.udp, verbose=True)
        self.heartbeat = HeartbeatMonitor(self.rank, self.world_size, self.udp, verbose=False)
        
        # Set callbacks
        self.gossip.set_failure_notify_callback(self._on_failure_detected)
        self.gossip.set_recovery_notify_callback(self._on_recovery_notify)
        self.heartbeat.set_failure_callback(self._on_heartbeat_failure)
        
        # TCP sockets
        self.sender_sock = None
        self.listener_sock = None
        self.sock_lock = threading.Lock()
        self.server_sock = None
        
        # Thread control
        self.stop_flag = False
        self.completion_event = None
        self.allreduce_active = threading.Event()  # Signals when allreduce is running

        # 3-way barrier for synchronization
        self.barrier_lock = threading.Lock()
        self.threads_at_barrier = 0
        self.barrier_cv = threading.Condition(self.barrier_lock)
        self.barrier_generation = 0

        # Setup ring topology
        self._initial_ring_setup()
        
        # If recovering, load checkpoint and notify others
        if self.is_recovering:
            self._log("I AM RECOVERING - Loading checkpoint and notifying cluster")
            checkpoint = self.checkpoint_mgr.load()
            if checkpoint:
                resume_iter = checkpoint['iteration']
                with self.iteration_lock:
                    self.iteration = resume_iter
                self._log(f"Loaded checkpoint at iteration {resume_iter}")
                
                # Notify others to resume at this iteration
                self.gossip.initiate_recovery_notify(resume_iter)
                time.sleep(3.0)  # Give gossip time to propagate
        
        # Connect TCP ring
        self._connect_tcp_ring()
        
        # Allreduce state
        self.chunks = []
        self.send_queue = queue.Queue()
        self.num_steps = 0
        self.max_payload_bytes = 0
        self.recv_buffer = None  # For framed protocol (bytearray)
        self.fast_recv_buffer = None  # For fast_net protocol (numpy float16)
        
        # Start worker threads
        self.listener_thread = threading.Thread(target=self._listener_loop, daemon=True)
        self.sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self.listener_thread.start()
        self.sender_thread.start()
        
        # Create initial checkpoint at iteration 0 (only if not recovering)
        if self.iteration == 0 and not self.is_recovering:
            self._log("Creating initial checkpoint at iteration 0")
            self._auto_checkpoint()
        
        self._log(f"Initialized (state={self.state.name}, iteration={self.iteration})")

    def _log(self, msg):
        """Log with timestamp and rank"""
        if self.verbose:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] [Rank {self.rank}] {msg}")

    # ========================================================================
    # CALLBACKS - Called by infrastructure
    # ========================================================================

    def _on_heartbeat_failure(self, failed_rank):
        """Heartbeat detected failure -> trigger gossip"""
        self._log(f"Heartbeat failure detected: rank {failed_rank}")
        self.gossip.initiate_failure_notify(failed_rank)
        self._transition_to_recovering()

    def _on_failure_detected(self, failed_rank, initiator_rank):
        """Gossip notified us of failure -> stop the world"""
        self._log(f"Failure notification: rank {failed_rank} failed (detected by rank {initiator_rank})")
        self._transition_to_recovering()

    def _on_recovery_notify(self, checkpoint_iter, initiator_rank):
        """Gossip notified us to resume -> rollback and resume"""
        self._log(f"Recovery notification: resume at iteration {checkpoint_iter} (from rank {initiator_rank})")
        
        # IMPORTANT: If we're still in NORMAL state (fast restart before failure detected),
        # we need to transition to RECOVERING first to stop ongoing operations
        with self.state_lock:
            if self.state == State.NORMAL:
                self._log("Fast restart detected - transitioning to RECOVERING first")
        self._transition_to_recovering()
        
        # Rollback model to checkpoint
        self._rollback_to_checkpoint(checkpoint_iter)
        
        # Reconnect TCP ring
        self._reconnect_tcp_ring()
        
        # Resume operation
        self._transition_to_normal(checkpoint_iter)

    # ========================================================================
    # STATE TRANSITIONS
    # ========================================================================

    def _transition_to_recovering(self):
        """Transition to RECOVERING state"""
        with self.state_lock:
            if self.state == State.RECOVERING:
                self._log("Already in RECOVERING state")
                return
            
            self._log(f"TRANSITION: {self.state.name} -> RECOVERING")
            self.state = State.RECOVERING
        
        # Stop-the-world actions
        self._log("STOP-THE-WORLD")
        self.heartbeat.pause_heartbeats()
        self._close_tcp_sockets()
        
        # Wake up any blocked allreduce
        # IMPORTANT: We set the event to wake up the main thread,
        # but the main thread MUST check self.state after waking up
        # to detect that this was a failure wake-up, not a success!
        if self.completion_event:
            self.completion_event.set()
        
        # Clear allreduce active flag
        self.allreduce_active.clear()
        
        # Reset barrier state
        with self.barrier_cv:
            self.threads_at_barrier = 0
            self.barrier_generation += 1
            self.barrier_cv.notify_all()

        # Flush queue (may not exist during init)
        if hasattr(self, 'send_queue'):
            try:
                while not self.send_queue.empty():
                    self.send_queue.get_nowait()
            except queue.Empty:
                pass

    def _transition_to_normal(self, checkpoint_iter):
        """Transition to NORMAL state"""
        with self.state_lock:
            self._log(f"TRANSITION: {self.state.name} -> NORMAL")
            self.state = State.NORMAL
        
        # Update iteration
        with self.iteration_lock:
            self.iteration = checkpoint_iter
            self._log(f"Iteration counter reset to {self.iteration}")
        
        # Reset CuPy RNG to prevent CURAND internal errors after recovery
        # Need to completely recreate the random state, not just reseed
        try:
            seed = 42 + self.rank * 1000 + checkpoint_iter
            # Create a fresh RandomState and set it as the global state
            new_rs = cp.random.RandomState(seed=seed)
            cp.random.set_random_state(new_rs)
            self._log(f"CuPy RNG recreated (seed={seed})")
        except Exception as e:
            self._log(f"CuPy RNG reset failed: {e}")
        
        # Resume infrastructure
        self.heartbeat.resume_heartbeats()
        self.heartbeat.reset_monitor()
        
        self._log("Resumed - ready for next allreduce")

    def _rollback_to_checkpoint(self, checkpoint_iter):
        """Rollback model state to checkpoint via callback"""
        self._log(f"Rolling back to iteration {checkpoint_iter}")
        
        # Try to load the specific iteration first
        checkpoint = self.checkpoint_mgr.load(checkpoint_iter)
        if not checkpoint:
            # Fallback: try to find closest available checkpoint
            available = self.checkpoint_mgr.get_available_iterations()
            self._log(f"Available checkpoints: {available}")
            
            # Find the closest iteration <= requested
            valid_iters = [i for i in available if i <= checkpoint_iter]
            if valid_iters:
                closest_iter = max(valid_iters)
                self._log(f"Using closest checkpoint: iteration {closest_iter}")
                checkpoint = self.checkpoint_mgr.load(closest_iter)
            else:
                self._log("No valid checkpoint found!")
                return
        
        # Restore model via callback!
        self.set_model_state_fn(checkpoint['model_state'])
        self._log("Model state restored via callback")
        
        # Restore RNG states
        self._set_rng_states(checkpoint.get('rng_states', {}))

    # ========================================================================
    # RING SETUP
    # ========================================================================

    def _initial_ring_setup(self):
        """Initial discovery and ring topology setup"""
        self._log("Discovering cluster topology...")
        
        # Get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        finally:
            s.close()

        # Start TCP server
        self.listen_port = BASE_PORT + self.rank
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_sock.bind(('0.0.0.0', self.listen_port))
        self.server_sock.listen(5)
        self._log(f"TCP server listening on port {self.listen_port}")

        # Register with discovery
        self.my_info = {
            "ip": local_ip,
            "data_port": self.listen_port,
            "udp_port": self.udp_port,
            "rank": self.rank,
            "world_size": self.world_size,
            "run_id": self.run_id
        }

        while True:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_disc:
                    s_disc.connect((DISCOVERY_IP, DISCOVERY_PORT))
                    serialized = pickle.dumps(self.my_info)
                    s_disc.sendall(struct.pack('>I', len(serialized)) + serialized)
                    all_clients = recv_all_pickle(s_disc)

                if not all_clients:
                    time.sleep(2)
                    continue
                
                if isinstance(all_clients, dict):
                    all_clients = list(all_clients.values())

                if len(all_clients) == self.world_size:
                    self._log(f"All {self.world_size} nodes discovered")
                    break
                else:
                    self._log(f"Waiting... ({len(all_clients)}/{self.world_size} nodes)")
                    time.sleep(2)
            except Exception as e:
                self._log(f"Discovery error: {e}")
                time.sleep(2)

        # Set up ring topology
        all_clients.sort(key=lambda x: x["rank"])
        my_index = next(i for i, c in enumerate(all_clients) if c["rank"] == self.rank)
        self.right_neighbor_info = all_clients[(my_index + 1) % self.world_size]
        self.left_neighbor_info = all_clients[(my_index - 1) % self.world_size]

        self._log(f"Ring: rank {self.left_neighbor_info['rank']} → me → rank {self.right_neighbor_info['rank']}")

        # Configure fault tolerance components
        self.heartbeat.set_neighbors(self.left_neighbor_info, self.right_neighbor_info)
        self.gossip.set_neighbors(self.left_neighbor_info, self.right_neighbor_info)

    def _connect_tcp_ring(self):
        """Connect TCP sockets in ring topology"""
        self._log("Connecting TCP ring...")
        
        accept_done = threading.Event()
        
        def accept_left():
            self.server_sock.settimeout(None)
            conn, addr = self.server_sock.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(SOCKET_RECV_TIMEOUT)
            with self.sock_lock:
                self.listener_sock = conn
            self._log(f"Accepted from left neighbor (rank {self.left_neighbor_info['rank']})")
            accept_done.set()
        
        threading.Thread(target=accept_left, daemon=True).start()

        # Connect to right neighbor
        target_ip = self.right_neighbor_info["ip"]
        target_port = self.right_neighbor_info["data_port"]
        
        while not self.stop_flag:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                sock.settimeout(2.0)
                sock.connect((target_ip, target_port))
                sock.settimeout(SOCKET_RECV_TIMEOUT)
                
                with self.sock_lock:
                    self.sender_sock = sock
                self._log(f"Connected to right neighbor (rank {self.right_neighbor_info['rank']})")
                break
            except (ConnectionRefusedError, socket.timeout, OSError):
                try: sock.close()
                except: pass
                time.sleep(1.0)

        accept_done.wait()
        
        with self.state_lock:
            current_state = self.state
        
        if current_state == State.RECOVERING:
            # **RECOVERY MODE**: Όλοι περιμένουν πρώτα (recv), μετά στέλνουν (send)
            # Αυτό σπάει το deadlock γιατί όλοι ξεκινούν από recv
            self._log("Recovery handshake: RECV -> SEND")
            try:
                data = self.listener_sock.recv(16)
                if data:
                    msg = data.decode("utf-8", errors="replace").strip()
                    self._log(f"Received: {msg}")
                else:
                    self._log("Connection closed during recv")
            except socket.timeout:
                self._log("Recv timeout during recovery handshake")
            except socket.error as e:
                self._log(f"Socket error during recv: {e}")
            
            # Τώρα στείλε
            self.sender_sock.sendall(b"READY\n")
            
        else:
            # **NORMAL MODE**: Αρχική σύνδεση - όλοι στέλνουν πρώτα, μετά περιμένουν
            self._log("Normal handshake: SEND -> RECV")
            self.sender_sock.sendall(b"READY\n")
            
            try:
                data = self.listener_sock.recv(16)
                if data:
                    msg = data.decode("utf-8", errors="replace").strip()
                    self._log(f"Received: {msg}")
                else:
                    self._log("Connection closed during recv")
            except socket.timeout:
                self._log("Recv timeout")
            except socket.error as e:
                self._log(f"Socket error: {e}")
        
        self._log("TCP ring connected")

    def _close_tcp_sockets(self):
        """Close TCP sockets"""
        with self.sock_lock:
            if self.sender_sock:
                try: 
                    self.sender_sock.close()
                    self._log("TCP sender socket closed")
                except: pass
                self.sender_sock = None
            if self.listener_sock:
                try: 
                    self.listener_sock.close()
                    self._log("TCP listener socket closed")
                except: pass
                self.listener_sock = None

    def _reconnect_tcp_ring(self):
        """Reconnect TCP ring after failure"""
        self._log("Reconnecting TCP ring...")
        
        # Poll discovery for updated topology
        updated = False
        while not updated and not self.stop_flag:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s_disc:
                    s_disc.connect((DISCOVERY_IP, DISCOVERY_PORT))
                    serialized = pickle.dumps(self.my_info)
                    s_disc.sendall(struct.pack('>I', len(serialized)) + serialized)
                    all_clients = recv_all_pickle(s_disc)
                
                if isinstance(all_clients, dict):
                    all_clients = list(all_clients.values())

                if len(all_clients) == self.world_size:
                    all_clients.sort(key=lambda x: x["rank"])
                    my_index = next(i for i, c in enumerate(all_clients) if c["rank"] == self.rank)
                    self.right_neighbor_info = all_clients[(my_index + 1) % self.world_size]
                    self.left_neighbor_info = all_clients[(my_index - 1) % self.world_size]
                    
                    self.heartbeat.set_neighbors(self.left_neighbor_info, self.right_neighbor_info)
                    self.gossip.set_neighbors(self.left_neighbor_info, self.right_neighbor_info)
                    updated = True
                    self._log(f"Topology updated: {self.left_neighbor_info['rank']} → me → {self.right_neighbor_info['rank']}")
                else:
                    time.sleep(1)
            except:
                time.sleep(1)

        # Reconnect
        self._connect_tcp_ring()
        self._log("TCP ring reconnected")

    # ========================================================================
    # THREAD SYNCHRONIZATION
    # ========================================================================

    def _barrier_sync(self, caller="unknown"):
        """
        3-way barrier: main, listener, sender threads synchronize here
        This ensures all threads finish the current allreduce before starting the next one
        """
        with self.barrier_cv:
            # Check if we're in recovery - don't wait at barrier
            if self.state == State.RECOVERING:
                return
            
            self.threads_at_barrier += 1
            current_gen = self.barrier_generation
            
            if self.threads_at_barrier >= 3:
                # All threads arrived - advance generation and wake everyone
                self.threads_at_barrier = 0
                self.barrier_generation += 1
                self.barrier_cv.notify_all()
            else:
                # Wait for other threads or state change
                while (self.barrier_generation == current_gen and 
                       self.threads_at_barrier < 3 and 
                       self.state == State.NORMAL):
                    self.barrier_cv.wait(timeout=1.0)

    # ========================================================================
    # WORKER THREADS
    # ========================================================================

    def _listener_loop(self):
        """Background thread: receive chunks from left neighbor"""
        while not self.stop_flag:
            try:
                # Wait for state to be NORMAL
                while self.state != State.NORMAL and not self.stop_flag:
                    continue
                
                if self.stop_flag:
                    break
                
                # Wait for allreduce to start
                if not self.allreduce_active.wait(timeout=1.0):
                    continue
                
                # Process ring allreduce steps
                for step in range(2 * self.num_steps - 2):
                    if self.state != State.NORMAL:
                        break
                    
                    recv_chunk_idx = (self.rank - step - 1) % self.world_size
                    
                    with self.sock_lock:
                        sock = self.listener_sock
                    
                    if not sock:
                        raise OSError("Socket is None")
                    
                    template = self.chunks[recv_chunk_idx]
                    
                    # Fast sockets (C extension) or fallback to pure Python
                    if HAS_FAST_NET:
                        _, received_data = recv_and_unpack_fast(sock, template, self.fast_recv_buffer)
                    else:
                        _, received_data = recv_and_unpack_framed(
                            sock, template, self.recv_buffer, timeout=SOCKET_RECV_TIMEOUT
                        )
                    
                    # Accumulate or replace
                    if step < self.num_steps - 1:
                        for i in range(len(self.chunks[recv_chunk_idx])):
                            self.chunks[recv_chunk_idx][i] += received_data[i]
                    else:
                        self.chunks[recv_chunk_idx] = received_data
                    
                    # Queue for sending
                    self.send_queue.put(recv_chunk_idx)
                
                # Signal completion
                if self.completion_event and self.state == State.NORMAL:
                    self.completion_event.set()
                
                # Barrier: wait for all threads before next allreduce
                self._barrier_sync("listener")
                
            except Exception as e:
                if self.state == State.RECOVERING:
                    self.allreduce_active.clear()
                    continue
                
                if "Socket closed" in str(e) or isinstance(e, (EOFError, ConnectionResetError, BrokenPipeError, OSError)):
                    # Don't trigger failure detection here - let heartbeat/gossip handle it
                    # Socket may have closed due to recovery transition, not actual failure
                    self.allreduce_active.clear()
                    time.sleep(0.5)
                    continue
                
                self._log(f"Listener error: {e}")
                self.allreduce_active.clear()
                time.sleep(1)

    def _sender_loop(self):
        """Background thread: send chunks to right neighbor"""
        while not self.stop_flag:
            try:
                # Wait for state to be NORMAL
                while self.state != State.NORMAL and not self.stop_flag:
                    continue
                
                if self.stop_flag:
                    break
                
                # Wait for allreduce to start
                if not self.allreduce_active.wait(timeout=1.0):
                    continue
                
                # Process ring allreduce steps
                for step in range(2 * self.num_steps - 2):
                    if self.state != State.NORMAL:
                        break

                    while True:
                        try:
                            chunk_idx = self.send_queue.get_nowait()
                            break
                        except queue.Empty:
                            if self.state == State.RECOVERING:
                                break
                            continue
                    
                    if self.state == State.RECOVERING:
                        break

                    with self.sock_lock:
                        sock = self.sender_sock
                    
                    if not sock:
                        raise OSError("Socket is None")
                    
                    with self.iteration_lock:
                        iter_id = self.iteration
                    
                    # Fast sockets (C extension) or fallback to pure Python
                    if HAS_FAST_NET:
                        send_chunk_fast(sock, self.chunks[chunk_idx])
                    else:
                        send_chunk_framed(sock, self.chunks[chunk_idx], 
                                        iteration_id=iter_id, chunk_id=chunk_idx)
                    
                    self.send_queue.task_done()

                # Barrier: wait for all threads before next allreduce
                self._barrier_sync("sender")

            except Exception as e:
                if self.state == State.RECOVERING:
                    continue
                
                if "Socket closed" in str(e) or isinstance(e, (EOFError, ConnectionResetError, BrokenPipeError, OSError)):
                    # Don't trigger failure detection here - let heartbeat/gossip handle it
                    # Socket may have closed due to recovery transition, not actual failure
                    time.sleep(0.5)
                    continue
                
                self._log(f"Sender error: {e}")
                time.sleep(1)

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def allreduce(self, data_list):
        """
        Perform fault-tolerant ring allreduce
        
        This is the ONLY method the application needs to call!
        All fault tolerance happens automatically:
        - Checkpointing (via get_model_state_fn)
        - Failure detection
        - Recovery coordination
        - State rollback (via set_model_state_fn)
        - Transparent retry
        
        Args:
            data_list: List of numpy/cupy arrays to average
        
        Returns:
            Tuple of (averaged_gradients, recovery_info) where recovery_info is a dict:
            - 'recovered': bool - whether recovery happened during this call
            - 'resume_iteration': int - iteration to resume from (if recovered)
        """

        recovery_happened = False

        while True:
            # Wait for recovery if needed
            while self.state == State.RECOVERING:
                recovery_happened = True
                time.sleep(0.1)
            
            if recovery_happened:
                with self.iteration_lock:
                    current_iter = self.iteration
                recovery_info = {
                    'recovered': recovery_happened,
                    'resume_iteration': current_iter
                }
                
                return None, recovery_info

            # Auto-checkpoint
            if self.iteration > 0 and self.iteration % self.checkpoint_interval == 0:
                self._auto_checkpoint()
            
            try:
                # Perform ring allreduce
                result = self._do_ring_allreduce(data_list)
                
                # Success! Increment iteration and return
                with self.iteration_lock:
                    current_iter = self.iteration
                    self.iteration += 1
                
                recovery_info = {
                    'recovered': recovery_happened,
                    'resume_iteration': current_iter
                }
                
                return result, recovery_info
                
            except Exception as e:
                # Any error -> wait for recovery and retry
                if self.state == State.RECOVERING:
                    self._log("AllReduce failed during recovery, will retry...")
                    recovery_happened = True
                    continue
                
                self._log(f"AllReduce error: {e}, waiting for recovery...")
                
                # Wait for recovery
                while self.state == State.RECOVERING:
                    recovery_happened = True
                    time.sleep(0.1)
                
                # Retry
                continue

    def _do_ring_allreduce(self, data_list):
        """Perform the actual ring allreduce operation"""
        # Prepare chunks
        self.chunks = chunk_list(data_list, self.world_size)
        self.num_steps = self.world_size

        # Allocate receive buffer
        self.max_payload_bytes = 0
        max_elements = 0
        for ch in self.chunks:
            bytes_for_ch = sum(int(np.array(arr).size) * np.dtype(np.float16).itemsize for arr in ch)
            elements_for_ch = sum(int(np.array(arr).size) for arr in ch)
            self.max_payload_bytes = max(self.max_payload_bytes, bytes_for_ch)
            max_elements = max(max_elements, elements_for_ch)

        # Framed protocol buffer (bytearray)
        if self.recv_buffer is None or len(self.recv_buffer) < self.max_payload_bytes:
            self.recv_buffer = bytearray(self.max_payload_bytes + 1024)
        
        # Fast_net protocol buffer (numpy float16) - reuse if large enough
        if self.fast_recv_buffer is None or len(self.fast_recv_buffer) < max_elements:
            self.fast_recv_buffer = np.empty(max_elements, dtype=np.float16)

        # Clear queue
        while not self.send_queue.empty():
            try: self.send_queue.get_nowait()
            except: break

        # Queue initial chunk
        my_chunk_idx = self.rank
        self.send_queue.put(my_chunk_idx)

        # Create completion event
        self.completion_event = threading.Event()

        # Signal threads to start
        self.allreduce_active.set()

        # Wait for completion
        if not self.completion_event.wait(timeout=60.0):
            self.allreduce_active.clear()
            raise TimeoutError("AllReduce timeout")

        # CRITICAL: Check if we're in recovery state
        # If failure happened during allreduce, event might have been set
        # by _transition_to_recovering() to wake us up, but data is incomplete!
        if self.state == State.RECOVERING:
            self.allreduce_active.clear()
            raise RuntimeError("AllReduce interrupted by failure")

        # Average results
        averaged_gradients = []
        for chunk in self.chunks:
            for arr in chunk:
                arr /= float(self.world_size)
                averaged_gradients.append(arr)

        self.allreduce_active.clear()

        # Barrier: wait for all threads before returning
        # This ensures listener/sender have finished before we modify chunks in next allreduce
        self._barrier_sync("main")

        # # IMPORTANT: Clear flag AFTER barrier so listener/sender see it's done
        # self.allreduce_active.clear()

        return averaged_gradients

    def _auto_checkpoint(self):
        """Automatically checkpoint via callback"""
        self._log(f"Auto-checkpointing iteration {self.iteration}")
        
        # Get model state via callback
        model_state = self.get_model_state_fn()
        
        # Get RNG states
        rng_states = self._get_rng_states()
        
        # Save
        elapsed = self.checkpoint_mgr.save(self.iteration, model_state, rng_states)
        self._log(f"Checkpoint saved in {elapsed:.3f}s")

    # ========================================================================
    # RNG STATE MANAGEMENT
    # ========================================================================

    def _get_rng_states(self):
        """Get all RNG states for reproducibility"""
        return {
            'std': random.getstate(),
            'numpy': np.random.get_state(),
            'cupy': cp.random.get_random_state()
        }

    def _set_rng_states(self, states):
        """Restore RNG states"""
        if not states:
            return
        try:
            random.setstate(states['std'])
            np.random.set_state(states['numpy'])
            cp.random.set_random_state(states['cupy'])
            self._log("RNG states restored")
        except Exception as e:
            self._log(f"Error restoring RNG: {e}")

    # ========================================================================
    # CLEANUP
    # ========================================================================

    def close(self):
        """Clean shutdown"""
        self._log("Shutting down...")
        
        self.stop_flag = True
        
        try:
            self.heartbeat.close()
        except:
            pass
        try:
            self.gossip.reset_state()
        except:
            pass
        try:
            self.udp.close()
        except:
            pass
        
        try:
            self.listener_thread.join(timeout=1)
        except:
            pass
        try:
            self.sender_thread.join(timeout=1)
        except:
            pass
        
        self._close_tcp_sockets()
        
        if self.server_sock:
            try:
                self.server_sock.close()
            except:
                pass
        
        self.checkpoint_mgr.delete()
        self._log("Shutdown complete")
