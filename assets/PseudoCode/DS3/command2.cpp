func erase(rt, x):
  if rt is null: return
  if x < rt.val:
    erase(rt.L, x)
  if x > rt.val:
    erase(rt.R, x)

  if x == rt.val:
    if (rt.R):
      val = getMin(rt.R)
      rt.val = val.val
      erase(rt.R, val.val)
    if not (rt.R):
      tmp = rt
      rt = rt.L
      free(tmp)

    if (unbalanced(rt)):
      rotate(rt)