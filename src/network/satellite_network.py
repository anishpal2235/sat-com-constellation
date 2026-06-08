"""
Iridium-like LEO Satellite Constellation Network Simulation.
Implements the satellite network topology, inter-satellite links,
UPF deployment, and dynamic topology changes.
"""

import numpy as np
from collections import defaultdict
import heapq
from config import Config


class Satellite:
    """Represents a single satellite in the constellation."""

    def __init__(self, sat_id, orbit_plane, index_in_plane, cpu_capacity):
        self.sat_id = sat_id
        self.orbit_plane = orbit_plane
        self.index_in_plane = index_in_plane
        self.cpu_capacity = cpu_capacity
        self.available_cpu = cpu_capacity
        self.has_upf = False
        self.upf_id = None
        self.neighbors = []  # List of (neighbor_sat_id, bandwidth)

    def reset(self):
        self.available_cpu = self.cpu_capacity


class SatelliteNetwork:
    """
    Simulates the Iridium-like LEO satellite constellation.
    66 satellites in 6 orbital planes (11 per plane).
    +Grid connection: each satellite connects to 2 intra-plane and 2 inter-plane neighbors.
    """

    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.num_satellites = Config.NUM_SATELLITES
        self.num_planes = Config.NUM_ORBIT_PLANES
        self.sats_per_plane = Config.SATS_PER_PLANE
        self.num_upfs = Config.NUM_UPFS

        self.satellites = {}
        self.adjacency = defaultdict(list)  # sat_id -> [(neighbor_id, bandwidth)]
        self.upf_satellites = []  # List of satellite IDs that have UPFs
        self.upf_to_sat = {}     # upf_id -> sat_id
        self.sat_to_upf = {}     # sat_id -> upf_id

        self._build_constellation()
        self._deploy_upfs()
        self._build_topology()

        # Floyd-Warshall shortest distances
        self.shortest_distances = None
        self._compute_shortest_distances()

    def _build_constellation(self):
        """Create 66 satellites in 6 orbital planes."""
        cpu_capacities = np.random.uniform(
            Config.CPU_SATELLITES_MIN,
            Config.CPU_SATELLITES_MAX,
            self.num_satellites
        )

        sat_id = 0
        for plane in range(self.num_planes):
            for idx in range(self.sats_per_plane):
                self.satellites[sat_id] = Satellite(
                    sat_id=sat_id,
                    orbit_plane=plane,
                    index_in_plane=idx,
                    cpu_capacity=cpu_capacities[sat_id]
                )
                sat_id += 1

    def _deploy_upfs(self):
        """Deploy 6 UPFs per orbital plane (36 total)."""
        upf_id = 0
        for plane in range(self.num_planes):
            plane_sats = [
                s for s in self.satellites.values()
                if s.orbit_plane == plane
            ]
            for i in range(Config.UPFS_PER_PLANE):
                sat = plane_sats[i]
                sat.has_upf = True
                sat.upf_id = upf_id
                self.upf_satellites.append(sat.sat_id)
                self.upf_to_sat[upf_id] = sat.sat_id
                self.sat_to_upf[sat.sat_id] = upf_id
                upf_id += 1

    def _build_topology(self, time_slot=0):
        """
        Build +Grid topology for Iridium constellation.
        Each satellite connects to:
        - 2 intra-plane neighbors (same orbit, adjacent)
        - 2 inter-plane neighbors (adjacent orbits, closest)
        """
        self.adjacency.clear()

        for sat_id, sat in self.satellites.items():
            plane = sat.orbit_plane
            idx = sat.index_in_plane

            # Intra-plane neighbors (circular within plane)
            prev_idx = (idx - 1) % self.sats_per_plane
            next_idx = (idx + 1) % self.sats_per_plane
            prev_sat = plane * self.sats_per_plane + prev_idx
            next_sat = plane * self.sats_per_plane + next_idx

            bw1 = np.random.uniform(Config.BANDWIDTH_MIN, Config.BANDWIDTH_MAX)
            bw2 = np.random.uniform(Config.BANDWIDTH_MIN, Config.BANDWIDTH_MAX)

            self._add_link(sat_id, prev_sat, bw1)
            self._add_link(sat_id, next_sat, bw2)

            # Inter-plane neighbors (adjacent orbits)
            left_plane = (plane - 1) % self.num_planes
            right_plane = (plane + 1) % self.num_planes

            left_sat = left_plane * self.sats_per_plane + idx
            right_sat = right_plane * self.sats_per_plane + idx

            # Apply time-dependent connectivity variation
            phase_shift = time_slot * 0.1
            connectivity_factor = 0.8 + 0.2 * np.cos(phase_shift + plane * 0.5)

            if connectivity_factor > 0.3:  # Link exists
                bw3 = np.random.uniform(Config.BANDWIDTH_MIN, Config.BANDWIDTH_MAX)
                bw4 = np.random.uniform(Config.BANDWIDTH_MIN, Config.BANDWIDTH_MAX)
                self._add_link(sat_id, left_sat, bw3)
                self._add_link(sat_id, right_sat, bw4)

    def _add_link(self, sat1, sat2, bandwidth):
        """Add a bidirectional link between two satellites."""
        existing = [n for n, _ in self.adjacency[sat1] if n == sat2]
        if not existing:
            self.adjacency[sat1].append((sat2, bandwidth))
            self.adjacency[sat2].append((sat1, bandwidth))

    def _compute_shortest_distances(self):
        """Compute shortest distances between all satellite pairs using Floyd-Warshall."""
        n = self.num_satellites
        self.shortest_distances = np.full((n, n), np.inf)
        np.fill_diagonal(self.shortest_distances, 0)

        for sat_id in range(n):
            for neighbor_id, bw in self.adjacency[sat_id]:
                sat = self.satellites[sat_id]
                neighbor = self.satellites[neighbor_id]
                if sat.orbit_plane == neighbor.orbit_plane:
                    dist = Config.INTRA_PLANE_DISTANCE
                else:
                    dist = Config.INTER_PLANE_DISTANCE
                self.shortest_distances[sat_id][neighbor_id] = dist

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if self.shortest_distances[i][k] + self.shortest_distances[k][j] < self.shortest_distances[i][j]:
                        self.shortest_distances[i][j] = self.shortest_distances[i][k] + self.shortest_distances[k][j]

    def get_link_bandwidth(self, sat1, sat2):
        """Get bandwidth between two adjacent satellites."""
        for neighbor, bw in self.adjacency[sat1]:
            if neighbor == sat2:
                return bw
        return 0.0

    def get_distance(self, sat1, sat2):
        """Get shortest distance between two satellites."""
        return self.shortest_distances[sat1][sat2]

    def bfs_shortest_delay_path(self, source, destination):
        """
        BFS-based shortest delay path from source to destination satellite.
        Returns (path, total_delay) where delay = transmission + propagation.
        """
        if source == destination:
            return [source], 0.0

        n = self.num_satellites
        dist = [float('inf')] * n
        dist[source] = 0
        prev = [-1] * n
        visited = [False] * n
        pq = [(0, source)]

        while pq:
            d, u = heapq.heappop(pq)
            if visited[u]:
                continue
            visited[u] = True

            if u == destination:
                break

            for neighbor, bw in self.adjacency[u]:
                if visited[neighbor]:
                    continue
                sat_u = self.satellites[u]
                sat_n = self.satellites[neighbor]
                if sat_u.orbit_plane == sat_n.orbit_plane:
                    link_dist = Config.INTRA_PLANE_DISTANCE
                else:
                    link_dist = Config.INTER_PLANE_DISTANCE
                prop_delay = link_dist / Config.SPEED_OF_LIGHT
                new_dist = d + prop_delay

                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = u
                    heapq.heappush(pq, (new_dist, neighbor))

        if dist[destination] == float('inf'):
            return [], float('inf')

        path = []
        node = destination
        while node != -1:
            path.append(node)
            node = prev[node]
        path.reverse()

        return path, dist[destination]

    def bfs_shortest_delay_path_with_transmission(self, source, destination,
                                                    packet_size):
        """
        Shortest delay path considering both transmission and propagation delay.
        packet_size in Mb, bandwidth in Gbps.
        """
        if source == destination:
            return [source], 0.0

        n = self.num_satellites
        dist = [float('inf')] * n
        dist[source] = 0
        prev = [-1] * n
        visited = [False] * n
        pq = [(0, source)]

        while pq:
            d, u = heapq.heappop(pq)
            if visited[u]:
                continue
            visited[u] = True

            if u == destination:
                break

            for neighbor, bw in self.adjacency[u]:
                if visited[neighbor]:
                    continue
                sat_u = self.satellites[u]
                sat_n = self.satellites[neighbor]
                if sat_u.orbit_plane == sat_n.orbit_plane:
                    link_dist = Config.INTRA_PLANE_DISTANCE
                else:
                    link_dist = Config.INTER_PLANE_DISTANCE

                trans_delay = (packet_size / 1000.0) / bw
                prop_delay = link_dist / Config.SPEED_OF_LIGHT
                total_link_delay = trans_delay + prop_delay

                new_dist = d + total_link_delay
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    prev[neighbor] = u
                    heapq.heappush(pq, (new_dist, neighbor))

        if dist[destination] == float('inf'):
            return [], float('inf')

        path = []
        node = destination
        while node != -1:
            path.append(node)
            node = prev[node]
        path.reverse()

        return path, dist[destination]

    def update_topology(self, time_slot):
        """Update the satellite network topology for a given time slot."""
        np.random.seed(Config.SEED + time_slot * 100)
        self._build_topology(time_slot)
        self._compute_shortest_distances()

    def get_upf_state_matrix(self):
        """
        Extract the state matrix M(t) for all UPFs.
        Each UPF u has feature vector: [c_u, b_sum_u, f_avg_u, g_swi_u]
        Returns: numpy array of shape (U, 4)
        """
        U = self.num_upfs
        state_matrix = np.zeros((U, 4))

        for upf_id in range(U):
            sat_id = self.upf_to_sat[upf_id]
            sat = self.satellites[sat_id]

            c_u = sat.available_cpu / sat.cpu_capacity

            total_bw = sum(bw for _, bw in self.adjacency[sat_id])
            max_possible_bw = 4 * Config.BANDWIDTH_MAX
            b_sum = total_bw / max_possible_bw if max_possible_bw > 0 else 0

            distances = []
            for other_upf in range(U):
                if other_upf != upf_id:
                    other_sat = self.upf_to_sat[other_upf]
                    d = self.get_distance(sat_id, other_sat)
                    if d < np.inf:
                        distances.append(d)
            f_avg = np.mean(distances) if distances else 0
            max_dist = Config.INTRA_PLANE_DISTANCE * self.sats_per_plane
            f_avg_norm = f_avg / max_dist if max_dist > 0 else 0

            g_swi = 1.0

            state_matrix[upf_id] = [c_u, b_sum, f_avg_norm, g_swi]

        return state_matrix

    def get_total_bandwidth_for_sat(self, sat_id):
        """Get total ISL bandwidth for a satellite."""
        return sum(bw for _, bw in self.adjacency[sat_id])

    def consume_resources(self, sat_id, cpu_amount):
        """Consume CPU resources on a satellite."""
        sat = self.satellites[sat_id]
        if sat.available_cpu >= cpu_amount:
            sat.available_cpu -= cpu_amount
            return True
        return False

    def release_resources(self, sat_id, cpu_amount):
        """Release CPU resources on a satellite."""
        sat = self.satellites[sat_id]
        sat.available_cpu = min(sat.available_cpu + cpu_amount, sat.cpu_capacity)

    def reset_resources(self):
        """Reset all satellite resources to full capacity."""
        for sat in self.satellites.values():
            sat.reset()

    def get_upf_satellite(self, upf_id):
        """Get the satellite ID where a UPF is deployed."""
        return self.upf_to_sat[upf_id]

    def get_all_upf_ids(self):
        """Get list of all UPF IDs."""
        return list(range(self.num_upfs))
