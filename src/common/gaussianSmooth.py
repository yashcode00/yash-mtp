import math

def gauss(sigma, x):
    exp_val = -1 * (x ** 2) / (2 * sigma ** 2)
    divider = math.sqrt(2 * math.pi * sigma ** 2)
    return (1 / divider) * math.exp(exp_val)

def gauss_kernel(samples, sigma):
    v = []

    double_center = False
    if samples % 2 == 0:
        double_center = True
        samples -= 1
    steps = (samples - 1) // 2
    step_size = (3 * sigma) / steps

    for i in range(steps, 0, -1):
        v.append(gauss(sigma, i * step_size * -1))

    v.append(gauss(sigma, 0))
    if double_center:
        v.append(gauss(sigma, 0))

    for i in range(1, steps + 1):
        v.append(gauss(sigma, i * step_size))

    # print()
    # print(f"The kernel contains {len(v)} entries: {' '.join(map(str, v))}")
    assert len(v) == samples

    return v

def gauss_smoothen(values, sigma, samples):
    out = []
    kernel = gauss_kernel(samples, sigma)
    sample_side = samples // 2
    value_idx = samples // 2 + 1
    ubound = len(values)

    for i in range(ubound):
        sample = 0
        sample_ctr = 0
        # print(f"Now at value {i}: ", end='')
        
        for j in range(i - sample_side, i + sample_side + 1):
            # print(j, end=' ')
            
            if 0 < j < ubound:
                sample_weight_index = sample_side + (j - i)
                # print(f"({sample_weight_index} [{kernel[sample_weight_index]}]) ", end='')
                sample += kernel[sample_weight_index] * values[j]
                sample_ctr += 1
        
        smoothed = sample / sample_ctr if sample_ctr != 0 else 0
        # print(f"S: {sample} C: {sample_ctr} V: {values[i]} SM: {smoothed}")
        out.append(smoothed)
    
    return out
