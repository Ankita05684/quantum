import numpy as np
from scipy import integrate, signal
import matplotlib.pyplot as plt

l = 1
x = np.linspace(-l, l, 1000)
plt.figure(figsize=(15, 20))

fsquare = signal.square(2 * x * np.pi)
fsaw = signal.sawtooth(2 * x * np.pi, 1)
ftri = signal.sawtooth(2 * x * np.pi, .5)

# NOTE: scipy.integrate.simps was removed in newer SciPy versions;
# use scipy.integrate.simpson instead (same trapezoid/Simpson-rule integration).
a0 = lambda f: integrate.simpson(f, x=x) / l
an = lambda f, n: integrate.simpson(f * np.cos(n * np.pi * x / l), x=x) / l
bn = lambda f, n: integrate.simpson(f * np.sin(n * np.pi * x / l), x=x) / l

i = 1
for f in [fsquare, fsaw, ftri]:
    z = int('31' + str(i))
    plt.subplot(z)
    for nupper in range(10, 60, 10):  # for different number of terms
        a0s = a0(f) / 2
        total = a0s
        for n in range(1, nupper + 1):
            ans = an(f, n) * np.cos(n * np.pi * x / l)
            bns = bn(f, n) * np.sin(n * np.pi * x / l)
            total += ans + bns
        # NOTE: this plot call was originally indented inside the inner "for n"
        # loop, so it fired on every n instead of once per nupper -- that drew
        # 10+20+30+40+50 = 150 overlapping lines per subplot, all with duplicate
        # labels. Moving it here (after the inner loop finishes) draws exactly
        # one curve per nupper value, as intended.
        plt.plot(x, total, label=str(nupper) + ' terms')
    plt.xlabel('x')
    plt.ylabel('fourier sum')
    plt.plot(x, f, 'k', lw=2, label='original')
    i += 1
    plt.legend()
    plt.grid(True)

plt.show()
