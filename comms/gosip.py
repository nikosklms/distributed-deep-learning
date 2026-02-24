#!/usr/bin/env python3
# gossip.py
# Gossip protocol for distributed failure notification and recovery coordination

import threading
import time
from enum import IntEnum
from comms.udp_reliable import ReliableUDP, MessageType

class GossipMessageType(IntEnum):
    """Gossip-specific message types (extends MessageType)"""
    FAILURE_NOTIFY = 2   # "Node X has failed"
    RECOVERY_NOTIFY = 3  # "Resume at checkpoint Y"

class GossipProtocol:
    """
    Ring-based gossip protocol for failure notification and recovery coordination
    
    Protocol:
    1. Right neighbor of failed node initiates FAILURE_NOTIFY
    2. Each node forwards message to right neighbor (ring propagation)
    3. Each node stops training and waits for recovery
    4. Failed node recovers, loads checkpoint, sends RECOVERY_NOTIFY
    5. Each node forwards RECOVERY_NOTIFY and resumes training
    """
    
    def __init__(self, rank, world_size, udp_layer, verbose=False):
        """
        Initialize gossip protocol
        
        Args:
            rank: This node's rank
            world_size: Total number of nodes
            udp_layer: ReliableUDP instance
            verbose: Enable debug logging
        """
        self.rank = rank
        self.world_size = world_size
        self.udp = udp_layer
        self.verbose = verbose
        
        # Neighbor information (set via set_neighbors)
        self.left_neighbor = None
        self.right_neighbor = None
        
        # Message tracking for deduplication
        self.seen_messages = set()  # Set of (msg_type, msg_id) tuples
        self.msg_lock = threading.Lock()
        
        # Callbacks
        self.failure_notify_callback = None   # Called when FAILURE_NOTIFY received
        self.recovery_notify_callback = None  # Called when RECOVERY_NOTIFY received
        
        # State tracking
        self.stopped = False  # Are we in "stopped" state waiting for recovery?
        self.state_lock = threading.Lock()
        
        # Register handlers
        self.udp.register_handler(MessageType.FAILURE_NOTIFY, self._handle_failure_notify)
        self.udp.register_handler(MessageType.RECOVERY_NOTIFY, self._handle_recovery_notify)
        
        if self.verbose:
            print(f"[Gossip {self.rank}] Protocol initialized")
    
    def set_neighbors(self, left_neighbor_info, right_neighbor_info):
        """
        Set neighbor information after ring setup
        
        Args:
            left_neighbor_info: Dict with 'rank', 'ip', 'udp_port'
            right_neighbor_info: Dict with 'rank', 'ip', 'udp_port'
        """
        self.left_neighbor = left_neighbor_info
        self.right_neighbor = right_neighbor_info
        
        if self.verbose:
            print(f"[Gossip {self.rank}] Neighbors set: left={left_neighbor_info['rank']}, right={right_neighbor_info['rank']}")
    
    def set_failure_notify_callback(self, callback):
        """
        Register callback for FAILURE_NOTIFY
        
        Args:
            callback: Function(failed_rank, initiator_rank) called when failure notification received
        """
        self.failure_notify_callback = callback
    
    def set_recovery_notify_callback(self, callback):
        """
        Register callback for RECOVERY_NOTIFY
        
        Args:
            callback: Function(checkpoint_iteration, initiator_rank) called when recovery notification received
        """
        self.recovery_notify_callback = callback
    
    def _generate_message_id(self, msg_type, initiator_rank, failed_rank=None, checkpoint_iter=None, timestamp=None):
        """
        Generate unique message ID for deduplication
        
        Args:
            msg_type: Type of message
            initiator_rank: Rank that started the gossip
            failed_rank: (for FAILURE_NOTIFY) rank of failed node
            checkpoint_iter: (for RECOVERY_NOTIFY) checkpoint iteration
            timestamp: (for RECOVERY_NOTIFY) timestamp to distinguish multiple recoveries at same iteration
        
        Returns:
            Tuple that uniquely identifies this message
        """
        if msg_type == MessageType.FAILURE_NOTIFY:
            return (msg_type, initiator_rank, failed_rank)
        elif msg_type == MessageType.RECOVERY_NOTIFY:
            # CRITICAL FIX: Include timestamp to allow multiple recoveries at same iteration
            # Without this, if node fails twice at iteration 0, second recovery is treated as duplicate!
            return (msg_type, initiator_rank, checkpoint_iter, timestamp)
        else:
            return (msg_type, initiator_rank)
    
    def _is_duplicate_message(self, msg_id):
        """Check if we've already seen this message (thread-safe)"""
        with self.msg_lock:
            if msg_id in self.seen_messages:
                return True
            self.seen_messages.add(msg_id)
            
            # Limit memory: keep only last 100 messages
            if len(self.seen_messages) > 100:
                # Remove oldest (note: this is approximate, but good enough)
                self.seen_messages = set(list(self.seen_messages)[-100:])
            
            return False
    
    def _forward_to_right_neighbor(self, msg_type, payload):
        """Forward a gossip message to right neighbor"""
        if not self.right_neighbor:
            if self.verbose:
                print(f"[Gossip {self.rank}] Cannot forward: right neighbor not set")
            return
        
        dest_addr = (self.right_neighbor['ip'], self.right_neighbor['udp_port'])
        
        # Use reliable send for gossip messages (critical information)
        self.udp.send_reliable(msg_type, payload, dest_addr)
        
        if self.verbose:
            print(f"[Gossip {self.rank}] Forwarded {msg_type.name} to rank {self.right_neighbor['rank']}")
    
    def _handle_failure_notify(self, payload, sender_addr):
        """
        Handle incoming FAILURE_NOTIFY message
        
        Payload format:
        {
            'failed_rank': <rank of failed node>,
            'initiator_rank': <rank that detected failure>,
            'timestamp': <time when failure detected>
        }
        """
        failed_rank = payload.get('failed_rank')
        initiator_rank = payload.get('initiator_rank')
        timestamp = payload.get('timestamp')
        
        # CRITICAL FIX: If I am the node that supposedly failed (i.e., I just restarted),
        # ignore this message. It's old news about my own death.
        if failed_rank == self.rank:
            if self.verbose:
                print(f"[Gossip {self.rank}] Ignoring FAILURE_NOTIFY for myself (rank {failed_rank}). I am alive now.")
            return

        # Generate message ID for deduplication
        msg_id = self._generate_message_id(MessageType.FAILURE_NOTIFY, initiator_rank, failed_rank)
        
        # Check for duplicate
        if self._is_duplicate_message(msg_id):
            if self.verbose:
                print(f"[Gossip {self.rank}] Duplicate FAILURE_NOTIFY (failed={failed_rank}), ignoring")
            return
        
        print(f"\n[Gossip {self.rank}] FAILURE_NOTIFY received!")
        print(f"[Gossip {self.rank}]   Failed node: rank {failed_rank}")
        print(f"[Gossip {self.rank}]   Detected by: rank {initiator_rank}")
        
        # Mark ourselves as stopped (waiting for recovery)
        with self.state_lock:
            self.stopped = True
        
        # Forward to right neighbor ONLY if right neighbor is NOT the failed node
        # (Don't try to forward to the dead node - gossip stops here)
        if self.right_neighbor and self.right_neighbor['rank'] != failed_rank:
            self._forward_to_right_neighbor(MessageType.FAILURE_NOTIFY, payload)
        else:
            if self.verbose:
                print(f"[Gossip {self.rank}] Not forwarding to failed node {failed_rank} - gossip complete")
        
        # Call user callback
        if self.failure_notify_callback:
            try:
                self.failure_notify_callback(failed_rank, initiator_rank)
            except Exception as e:
                print(f"[Gossip {self.rank}] Error in failure_notify_callback: {e}")
    
    def _handle_recovery_notify(self, payload, sender_addr):
        """
        Handle incoming RECOVERY_NOTIFY message
        
        Payload format:
        {
            'checkpoint_iteration': <iteration to resume from>,
            'initiator_rank': <rank of recovered node>,
            'timestamp': <time when recovery started>
        }
        """
        checkpoint_iter = payload.get('checkpoint_iteration')
        initiator_rank = payload.get('initiator_rank')
        timestamp = payload.get('timestamp')
        
        # Generate message ID for deduplication
        # CRITICAL: Include timestamp so multiple recoveries at same iteration aren't treated as duplicates
        msg_id = self._generate_message_id(
            MessageType.RECOVERY_NOTIFY, 
            initiator_rank, 
            checkpoint_iter=checkpoint_iter,
            timestamp=timestamp
        )
        
        # Check for duplicate
        if self._is_duplicate_message(msg_id):
            if self.verbose:
                print(f"[Gossip {self.rank}] Duplicate RECOVERY_NOTIFY (iter={checkpoint_iter}, ts={timestamp}), ignoring")
            return
        
        print(f"\n[Gossip {self.rank}] RECOVERY_NOTIFY received!")
        print(f"[Gossip {self.rank}]   Resume at iteration: {checkpoint_iter}")
        print(f"[Gossip {self.rank}]   Initiated by: rank {initiator_rank}")
        print(f"[Gossip {self.rank}]   Timestamp: {timestamp}")
        
        # Mark ourselves as no longer stopped
        with self.state_lock:
            self.stopped = False
        
        # For RECOVERY, forward to ALL neighbors (complete the ring)
        # Unlike FAILURE_NOTIFY, we want the message to complete the full circle
        # because all nodes (including recovered node) need to coordinate
        self._forward_to_right_neighbor(MessageType.RECOVERY_NOTIFY, payload)
        
        # Call user callback
        if self.recovery_notify_callback:
            try:
                self.recovery_notify_callback(checkpoint_iter, initiator_rank)
            except Exception as e:
                print(f"[Gossip {self.rank}] Error in recovery_notify_callback: {e}")
    
    def initiate_failure_notify(self, failed_rank):
        """
        Initiate FAILURE_NOTIFY gossip (called by right neighbor of failed node)
        
        The initiator (right neighbor of failed node) does NOT stop itself.
        It stays active to detect when the failed node recovers.
        
        Args:
            failed_rank: Rank of the failed node
        """
        print(f"\n[Gossip {self.rank}] INITIATING FAILURE_NOTIFY for rank {failed_rank}")
        print(f"[Gossip {self.rank}] I will stay active and wait for node {failed_rank} to recover")
        
        payload = {
            'failed_rank': failed_rank,
            'initiator_rank': self.rank,
            'timestamp': time.time()
        }
        
        # Mark message as seen (so we don't process our own message when it comes back)
        msg_id = self._generate_message_id(MessageType.FAILURE_NOTIFY, self.rank, failed_rank)
        with self.msg_lock:
            self.seen_messages.add(msg_id)
        
        # Initiator does NOT mark itself as stopped - stays active!
        # with self.state_lock:
        #     self.stopped = True
        
        # Send to right neighbor
        # CRITICAL: Handle 2-node ring case where right neighbor IS the failed node
        if self.right_neighbor and self.right_neighbor['rank'] == failed_rank:
            print(f"[Gossip {self.rank}] Right neighbor is the failed node (2-node ring case).")
            print(f"[Gossip {self.rank}] I must also stop and wait for recovery.")
            
            # In 2-node ring, the initiator IS the only surviving node
            # It must also enter the waiting state and close TCP
            with self.state_lock:
                self.stopped = True
            
            # Call our own failure callback to trigger stop-the-world
            if self.failure_notify_callback:
                try:
                    self.failure_notify_callback(failed_rank, self.rank)
                except Exception as e:
                    print(f"[Gossip {self.rank}] Error in failure_notify_callback: {e}")
        else:
            self._forward_to_right_neighbor(MessageType.FAILURE_NOTIFY, payload)
            print(f"[Gossip {self.rank}] FAILURE_NOTIFY sent to others. I remain active.")
    
    def initiate_recovery_notify(self, checkpoint_iteration):
        """
        Initiate RECOVERY_NOTIFY gossip (called by recovered node)
        
        The recovered node sends the notification, then immediately starts working.
        It does NOT wait for the message to come back around the ring.
        
        Args:
            checkpoint_iteration: Iteration number to resume from
        """
        # Generate unique timestamp for THIS recovery attempt
        recovery_timestamp = time.time()
        
        print(f"\n[Gossip {self.rank}] INITIATING RECOVERY_NOTIFY (checkpoint iter={checkpoint_iteration})")
        print(f"[Gossip {self.rank}] Notifying others to resume, then I will start working")
        print(f"[Gossip {self.rank}] Recovery timestamp: {recovery_timestamp}")
        
        payload = {
            'checkpoint_iteration': checkpoint_iteration,
            'initiator_rank': self.rank,
            'timestamp': recovery_timestamp  # CRITICAL: Unique timestamp for each recovery
        }
        
        # Mark message as seen (with timestamp!)
        msg_id = self._generate_message_id(
            MessageType.RECOVERY_NOTIFY, 
            self.rank, 
            checkpoint_iter=checkpoint_iteration,
            timestamp=recovery_timestamp
        )
        with self.msg_lock:
            self.seen_messages.add(msg_id)
        
        # Initiator marks itself as not stopped (ready to work)
        with self.state_lock:
            self.stopped = False
        
        # Send to right neighbor
        self._forward_to_right_neighbor(MessageType.RECOVERY_NOTIFY, payload)
        
        # Initiator does NOT call its own callback - it just starts working immediately
        # Other nodes will call their callbacks when they receive the message
        
        print(f"[Gossip {self.rank}] RECOVERY_NOTIFY sent. I can now resume training.")
    
    def is_stopped(self):
        """Check if we're in stopped state (waiting for recovery)"""
        with self.state_lock:
            return self.stopped
    
    def reset_state(self):
        """Reset gossip state (useful for testing)"""
        with self.state_lock:
            self.stopped = False
        with self.msg_lock:
            self.seen_messages.clear()


# ============================================================================
# TESTING CODE
# ============================================================================

def test_gossip_protocol():
    """Test gossip protocol with 4 nodes in a ring"""
    print("\n" + "="*60)
    print("Testing Gossip Protocol")
    print("="*60 + "\n")
    
    # Create 4 nodes in a ring
    udp0 = ReliableUDP(rank=0, listen_port=9000, verbose=False)
    udp1 = ReliableUDP(rank=1, listen_port=9001, verbose=False)
    udp2 = ReliableUDP(rank=2, listen_port=9002, verbose=False)
    udp3 = ReliableUDP(rank=3, listen_port=9003, verbose=False)
    
    gossip0 = GossipProtocol(rank=0, world_size=4, udp_layer=udp0, verbose=True)
    gossip1 = GossipProtocol(rank=1, world_size=4, udp_layer=udp1, verbose=True)
    gossip2 = GossipProtocol(rank=2, world_size=4, udp_layer=udp2, verbose=True)
    gossip3 = GossipProtocol(rank=3, world_size=4, udp_layer=udp3, verbose=True)
    
    # Set up ring: 0 -> 1 -> 2 -> 3 -> 0
    gossip0.set_neighbors(
        left_neighbor_info={'rank': 3, 'ip': '127.0.0.1', 'udp_port': 9003},
        right_neighbor_info={'rank': 1, 'ip': '127.0.0.1', 'udp_port': 9001}
    )
    gossip1.set_neighbors(
        left_neighbor_info={'rank': 0, 'ip': '127.0.0.1', 'udp_port': 9000},
        right_neighbor_info={'rank': 2, 'ip': '127.0.0.1', 'udp_port': 9002}
    )
    gossip2.set_neighbors(
        left_neighbor_info={'rank': 1, 'ip': '127.0.0.1', 'udp_port': 9001},
        right_neighbor_info={'rank': 3, 'ip': '127.0.0.1', 'udp_port': 9003}
    )
    gossip3.set_neighbors(
        left_neighbor_info={'rank': 2, 'ip': '127.0.0.1', 'udp_port': 9002},
        right_neighbor_info={'rank': 0, 'ip': '127.0.0.1', 'udp_port': 9000}
    )
    
    # Track callbacks
    failure_notifications = []
    recovery_notifications = []
    
    def failure_callback(failed_rank, initiator_rank):
        failure_notifications.append((failed_rank, initiator_rank))
        print(f"*** CALLBACK: Node detected failure of rank {failed_rank} ***")
    
    def recovery_callback(checkpoint_iter, initiator_rank):
        recovery_notifications.append((checkpoint_iter, initiator_rank))
        print(f"*** CALLBACK: Node received recovery at iter {checkpoint_iter} ***")
    
    for g in [gossip0, gossip1, gossip2, gossip3]:
        g.set_failure_notify_callback(failure_callback)
        g.set_recovery_notify_callback(recovery_callback)
    
    # Test 1: FAILURE_NOTIFY propagation
    print("\n--- Test 1: FAILURE_NOTIFY Propagation ---")
    print("Simulating: Node 2 fails, Node 3 detects it\n")
    
    # Node 3 (right neighbor of node 2) initiates failure notify
    gossip3.initiate_failure_notify(failed_rank=2)
    
    # Wait for gossip to propagate around ring
    time.sleep(3)
    
    # Only nodes 0 and 1 should receive callbacks (they stop)
    # Node 3 (initiator) does NOT get callback - it stays active
    # Node 2 (failed) does not receive message
    assert len(failure_notifications) == 2, f"Expected 2 failure notifications, got {len(failure_notifications)}"
    assert all(failed_rank == 2 for failed_rank, _ in failure_notifications), "All should report rank 2 failed"
    
    # Only nodes 0 and 1 should be stopped
    # Node 3 stays active (to detect recovery)
    assert gossip0.is_stopped(), "Node 0 should be stopped"
    assert gossip1.is_stopped(), "Node 1 should be stopped"
    assert not gossip2.is_stopped(), "Node 2 (failed) never received notification"
    assert not gossip3.is_stopped(), "Node 3 (initiator) should stay active"
    
    print("Test 1 passed: FAILURE_NOTIFY propagated to all nodes\n")
    
    # Test 2: RECOVERY_NOTIFY propagation
    print("\n--- Test 2: RECOVERY_NOTIFY Propagation ---")
    print("Simulating: Node 2 recovers and broadcasts checkpoint\n")
    
    # Node 2 (recovered) initiates recovery notify
    gossip2.initiate_recovery_notify(checkpoint_iteration=100)
    
    # Wait for gossip to propagate
    time.sleep(3)
    
    # Nodes 3, 0, 1 should receive callbacks (they resume)
    # Node 2 (initiator) does NOT get callback - it just starts working
    assert len(recovery_notifications) == 3, f"Expected 3 recovery notifications, got {len(recovery_notifications)}"
    assert all(iter == 100 for iter, _ in recovery_notifications), "All should report iteration 100"
    
    # All nodes should be unstoppped now
    assert not gossip0.is_stopped(), "Node 0 should be running"
    assert not gossip1.is_stopped(), "Node 1 should be running"
    assert not gossip2.is_stopped(), "Node 2 should be running"
    assert not gossip3.is_stopped(), "Node 3 should be running"
    
    print("Test 2 passed: RECOVERY_NOTIFY propagated to all nodes\n")
    
    # Test 3: Multiple recoveries at same iteration (the bug you found!)
    print("\n--- Test 3: Multiple Recoveries at Same Iteration ---")
    print("Simulating: Node 2 crashes AGAIN and recovers at iter 100\n")
    
    recovery_count_before = len(recovery_notifications)
    
    # Simulate another crash at same iteration
    gossip3.initiate_failure_notify(failed_rank=2)
    time.sleep(2)
    
    # Node 2 recovers AGAIN at iteration 100 (second recovery attempt)
    gossip2.initiate_recovery_notify(checkpoint_iteration=100)
    time.sleep(3)
    
    # Should receive NEW callbacks (not deduplicated) because timestamp is different!
    assert len(recovery_notifications) > recovery_count_before, "Second recovery should NOT be deduplicated"
    print(f"Test 3 passed: Second recovery accepted (got {len(recovery_notifications) - recovery_count_before} new notifications)\n")
    
    # Test 4: Deduplication still works for identical messages
    print("\n--- Test 4: Message Deduplication ---")
    failure_count_before = len(failure_notifications)
    
    # Try to send same failure notify again (same timestamp)
    gossip3.initiate_failure_notify(failed_rank=2)
    time.sleep(2)
    
    # Should NOT receive new callbacks (deduplicated)
    assert len(failure_notifications) == failure_count_before, "Duplicate messages should be ignored"
    print("Test 4 passed: Duplicate messages ignored\n")
    
    # Cleanup
    print("\n--- Cleanup ---")
    udp0.close()
    udp1.close()
    udp2.close()
    udp3.close()
    
    print("\n" + "="*60)
    print("All tests passed!")
    print(f"Total failure notifications: {len(failure_notifications)}")
    print(f"Total recovery notifications: {len(recovery_notifications)}")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_gossip_protocol()