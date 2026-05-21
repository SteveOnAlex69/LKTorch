func search():
  sort(edges)
  dsu = new DSU()
  ans = []
  rejected = []
  
  for e in edges:
    if dsu.same_set(e.u, e.v):
      rejected.push(e);
    else:
      ans.push(e)
      dsu.join_set(e.u, e.v)
      
  return ans, rejected;