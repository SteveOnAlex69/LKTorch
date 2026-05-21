func search(rt, x):
  if rt is null:
    return null
  if rt.val == x:
    return rt
  if x < rt.val:
    return search(rt.L, x)
  if x >= rt.val:
    return search(rt.R, x)