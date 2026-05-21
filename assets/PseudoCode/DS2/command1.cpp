func insert(x):
  idx = hash(x)
  tmp = head[idx].next
  node = new Node(x)
  head[idx].next = node
  node.next = tmp