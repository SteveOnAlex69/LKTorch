func search(x):
  idx = hash(x)
  cur = head[idx].next
  while cur:
    if cur.val == x:
      return cur
    cur = cur.next
  return null