func insert(x):
  tmp = head.next
  node = new Node(x)
  head.next = node
  node.next = tmp