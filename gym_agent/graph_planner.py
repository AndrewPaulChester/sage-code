import pickle
import numpy as np
from collections import defaultdict
import networkx as nx
import operator


def bfs(start, goal, edges):

    if start == goal:
        return []
    frontier = [(start, [])]
    visited = []

    while frontier:
        current, path = frontier.pop(0)
        visited.append(current)
        for (action, destination) in edges[current]:
            if destination == goal:
                return path + [(current, action)]
            if destination not in visited + frontier:
                frontier.append((destination, path + [(current, action)]))

    return None


def edge_bfs(start, goals, edges):
    frontier = [(start, [])]
    visited = []

    while frontier:
        current, path = frontier.pop(0)
        visited.append(current)
        for (action, destination) in edges[current]:
            if goals[current, action]:
                return path + [(current, action)]
            if destination not in visited + frontier:
                frontier.append((destination, path + [(current, action)]))

    return None


class Planner:
    def __init__(self, transitions):
        self.jumps = self.precalc_jumps(transitions)
        self.action_lookup = self.precalc_action_lookup(transitions)
        self.maps = self.precalc_maps(transitions)
        self.sparse_maps = self.precalc_sparse_maps(transitions)
        self.relevant_maps = self.precalc_relevant_maps(transitions)
        self.shortest_paths = self.precalc_shortest_paths(transitions)
        self.self_loops = self.precalc_self_loops(transitions)
        self.reachable = self.precalc_reachable()
        self.any_unseen = True

    def precalc_jumps(self, transitions):
        return [
            (s, a, s2)
            for s, a, s2 in transitions
            if s % 20 != s2 % 20 and (s < 500 or s2 < 500)
        ]

    def precalc_action_lookup(self, transitions):
        # this doesn't work for self loops as there are multiple possible edges
        action_lookup = {}
        for (s, a, s2) in transitions:
            action_lookup[(s, s2)] = a
        return action_lookup

    def precalc_maps(self, transitions):
        maps = []
        for i in range(20):
            v = list(range(i, 500, 20))
            edges = [(s, a, s2) for s, a, s2 in transitions if s in v and s2 in v]
            e = defaultdict(list)
            for s, a, s2 in edges:
                e[s] += [(a, s2)]
            s, a, s2 = self.jumps[i]
            e[s] += [(a, s2)]
            maps.append((v, e))
        v = list(range(500, 525))
        edges = [(s, a, s2) for s, a, s2 in transitions if s in v and s2 in v]
        e = defaultdict(list)
        for s, a, s2 in edges:
            e[s] += [(a, s2)]
        maps.append((v, e))
        return maps

    def precalc_sparse_maps(self, transitions):
        maps = []
        for i in range(20):
            v = list(range(i, 500, 20))
            e = [
                (s, a, s2) for s, a, s2 in transitions if s in v and s2 in v and s != s2
            ]
            e += [self.jumps[i]]
            maps.append((v, e))
        v = list(range(500, 525))
        e = [(s, a, s2) for s, a, s2 in transitions if s in v and s2 in v and s != s2]
        maps.append((v, e))
        return maps

    def precalc_relevant_maps(self, transitions):
        relevant_maps = {}
        for i in range(525):
            relevant = [20]
            if i < 500:
                for j in range(4):
                    if i % 4 == j:
                        relevant.append(j + 16)
                mod = i % 20
                if mod < 16:
                    relevant.append(mod)

            relevant_maps[i] = list(reversed(relevant))
        return relevant_maps

    def precalc_shortest_paths(self, transitions):
        shortest_paths = {}
        for start in range(500):
            current_map = self.relevant_maps[start][0]
            goal, action, destination = self.jumps[current_map]
            shortest_path = bfs(start, goal, self.maps[current_map][1])
            shortest_paths[start] = shortest_path + [(goal, action)]
        return shortest_paths

    def precalc_self_loops(self, transitions):
        self_loops = [([], []) for _ in range(21)]
        for s, a, s2 in transitions:
            if s == s2:
                self_loops[self.relevant_maps[s][0]][0].append(s)
                self_loops[self.relevant_maps[s][0]][1].append(a)
        return self_loops

    def precalc_reachable(self):
        reachable = {}
        for start in range(525):
            r = []
            for m in self.relevant_maps[start]:
                r.extend(self.maps[m][0])
            reachable[start] = r
        return reachable

    def plan(self, state, threshold, rho, unseen):
        maps = self.relevant_maps[state]
        if self.any_unseen:
            if np.any(unseen[self.reachable[state]]):
                start = state
                path = []
                for m in maps:
                    v, e = self.maps[m]
                    if np.any(unseen[v]):
                        path.extend(edge_bfs(start, unseen, e))
                        return path
                    else:
                        path.extend(self.shortest_paths[start])
                        start = self.jumps[m][2]
                return path
            else:
                if not np.any(unseen):
                    self.any_unseen = False

        bonus = 0
        path_so_far = []
        start = state
        for m in maps:
            max_self_loop = self.get_max_self_loop(m, rho)
            if max_self_loop is not None:
                return max_self_loop
            qualities, paths = self.bellman_ford(start, m, rho)
            best_quality, best_path = self.get_best_nonempty_path(
                start, rho, qualities, paths
            )
            if -best_quality + bonus > threshold:
                path_so_far.extend(best_path)
                return path_so_far
            else:
                if start < 500:
                    start = self.jumps[m][2]
                    bonus -= qualities[start]
                    path_so_far.extend(self.retrieve_path(paths[start]))

        return None

    def get_max_self_loop(self, m, rho):
        if rho[self.self_loops[m]].max() > 0:
            print("positive self loop detected")
            raise ValueError
        else:
            return None

    def generate_network(self, m, rho):
        G = nx.DiGraph()
        G.add_weighted_edges_from(
            [(s, s2, -rho[s, a]) for s, a, s2 in self.sparse_maps[m][1]]
        )
        return G

    def bellman_ford(self, state, m, rho):
        G = self.generate_network(m, rho)
        qualities, paths = nx.single_source_bellman_ford(G, state)
        return qualities, paths

    def get_best_path(self, qualities, paths):
        dest, quality = min(qualities.items(), key=operator.itemgetter(1))
        path = self.retrieve_path(paths[dest])
        return quality, path

    def retrieve_path(self, path):
        our_path = []
        s = path[0]
        for i in range(1, len(path)):
            s2 = path[i]
            our_path.append((s, self.action_lookup[(s, s2)]))
            s = s2
        return our_path

    def get_best_nonempty_path(self, start, rho, qualities, paths):
        qualities[start] = 100000
        q, p = self.get_best_path(qualities, paths)
        action = np.argmax(rho[start])
        if -q > rho[start, action]:
            return q, p
        else:
            return -rho[start, action], [(start, action)]

if __name__ == "__main__":

    with open("data/transitions.pickle", "rb") as handle:
        transitions = pickle.load(handle)
    p = Planner(transitions)
