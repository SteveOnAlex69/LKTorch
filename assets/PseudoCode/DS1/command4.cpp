func update(x, y):
  cur = head.next
  while cur:
    if cur.val == x:
      cur.val = y
      return
    cur = cur.next