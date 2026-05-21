func erase(s):
  path = [root]
  cur = root
  for c in s:
    idx = c - '0'
    cur = cur.ch[idx]
    path.push(cur)
    if cur is null: 
      return

  cur.cnt--
  if cur.cnt > 0: 
    return

  for i = len(s) down to 1:
    if hasChild(path[i]):
      break
    free(path[i])
    p = path[i-1]
    idx = s[i-1] - '0'
    p.ch[idx] = null