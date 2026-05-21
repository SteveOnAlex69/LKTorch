func search(s):
  cur = root
  for c in s:
    if cur is null:
      return null
    idx = c - '0'
    cur = cur.ch[idx]
  return cur