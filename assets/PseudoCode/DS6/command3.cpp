func search(s, t):
  dis = array(INF)
  pq = MinHeap()
  
  dis[s] = 0
  pq.push((0, s))
  
  while not pq.empty():
    d, u = pq.pop()
    if u == t:
      return d
    for v, w in adj[u]:
      if d+w < dis[v]:
        dis[v] = d+w
        pq.push((d+w, v))
        
  return INF