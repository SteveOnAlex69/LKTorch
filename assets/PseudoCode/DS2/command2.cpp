func erase(x):
  idx = hash(x)
  cur = head[idx]
  while cur.next:
    nxt = cur.next
    if nxt.val == x:
      cur.next = nxt.next
      free(nxt)
      return
    cur = nxt