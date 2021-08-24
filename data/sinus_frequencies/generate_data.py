import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


np.random.seed(66)


def sinus_funcion(f, A, t):
    return A * np.sin(2*np.pi*f*(t + 0.005 * np.random.randn(len(t)))) + 0.01*A*np.random.randn(len(t))


Amp = 1.
freqs = [0.05, 0.1, 0.5, 1.0, 2.0]
L = 400
N = 25000
times = np.linspace(0, 6, L, dtype=np.float32)
X = np.zeros((N, L), dtype=np.float32)
Y = np.zeros((N, 1), dtype=np.float32)
for n in range(N):
    i = np.random.choice(np.arange(len(freqs)))
    freq = freqs[i]
    X[n, :] = sinus_funcion(freq, Amp, times)
    Y[n, :] = i

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

np.savetxt('./sinus_frequencies_X_TRAIN', X_train, delimiter=',')
np.savetxt('./sinus_frequencies_Y_TRAIN', Y_train, delimiter=',')
np.savetxt('./sinus_frequencies_X_TEST', X_test, delimiter=',')
np.savetxt('./sinus_frequencies_Y_TEST', Y_test, delimiter=',')

npoints = 128
plt.plot(X[:npoints].T)
print(Y[:npoints])
plt.show()
