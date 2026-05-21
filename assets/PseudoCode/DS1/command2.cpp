func erase(x):
  cur = head
  while cur.next:
    nxt = cur.next
    if nxt.val == x:
      cur.next = nxt.next
      delete(nxt)
      break
    cur = nxt