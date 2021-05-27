class MultiQueue:
    def __init__(self, n):
        self.n = n
        self.reset()

    def add_item(self, item, i):
        self.queues[i].append(item)

    def check_layer(self):
        if min([len(q) for q in self.queues]) > 0:
            return True
        return False

    def pop_layer(self):
        if min([len(q) for q in self.queues]) == 0:
            raise AssertionError("At least one queue is empty, cannot pop layer")

        return [q.pop(0) for q in self.queues]

    def reset(self):
        self.queues = [[] for _ in range(self.n)]
