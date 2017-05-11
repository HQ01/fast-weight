def receptive_field(operators):
  if not operators: return 1

  field = receptive_field(operators[:-1])
  kernel, _ = operators[-1][0]
  if len(operators) > 1: _, stride = operators[-2][0]
  else: stride = 1

  return field + (kernel - 1) * stride

operators = (
  ((5, 5), (1, 1)),
  ((2, 2), (2, 2)),
) * 3 + (
  ((5, 5), (1, 1)),
) * 2
print receptive_field(operators)
