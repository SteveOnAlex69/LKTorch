func search(x):
  cur = head.next
  while cur:
    if cur.val == x:
      return cur
    cur = cur.next
  return null