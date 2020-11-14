from queue import PriorityQueue

class Node(object):
    node_id = 0
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right
        self.node_id = Node.node_id
        Node.node_id += 1

    def children(self):
        return (self.left, self.right)

    def nodes(self):
        return (self.left, self.right)

freq = {3: 1, 1: 2, 2: 3, 5: 3, 6: 3, 4: 4}

q = PriorityQueue()
for item in freq.items():
    q.put((item[1], item[0]))

while q.qsize() > 1:
    l, r = q.get(), q.get()
    print(l, r)
    node = Node(l[1], r[1])
    q.put((l[0] + r[0], r[0]))