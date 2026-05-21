func insert(s):
  cur = root
  for c in s:
    idx = c - '0'
    if cur.ch[idx] is null:
      cur.ch[idx] = new Node()
    cur = cur.ch[idx]
  cur.cnt++