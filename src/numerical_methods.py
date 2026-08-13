import numpy as np

def calculate_mean(S):
    return np.sum(S) / S.shape[0]

def calculate_median(S):
    S_sorted = np.sort(S)
    return S_sorted[10]  # Hardcoded based on original logic

def calculate_std_dev(S, mean):
    variance = np.sum((S - mean)**2) / S.shape[0]
    return np.sqrt(variance)

def basis_function(X, k, x):
    fi = 1
    for i in range(X.shape[0]):
        if i != k:
            fi *= (x - X[i])
    return fi

def lagrange_interpolation(X, Y):
    A = np.zeros(X.shape[0])
    for i in range(A.shape[0]):
        A[i] = Y[i] / basis_function(X, i, X[i])
    return A

def lagrange_function(A, x, X):
    result = 0
    for i in range(A.shape[0]):
        result += A[i] * basis_function(X, i, x)
    return result

def gaussian_elimination(A, B):
    mC = np.zeros((A.shape[0], A.shape[1] + 1))
    wX = np.zeros(A.shape[0])
    mC[0:A.shape[0], 0:A.shape[0]] = A
    B = np.reshape(B, (A.shape[0], 1))
    mC[:, A.shape[0]:] = B
    n = mC.shape[0]

    for s in range(n - 1):
        for i in range(s + 1, n):
            for j in range(s + 1, n + 1):
                mC[i][j] = mC[i][j] - (mC[i][s] / mC[s][s]) * mC[s][j]

    wX[n - 1] = mC[n - 1][n] / mC[n - 1][n - 1]

    for i in range(n - 2, -1, -1):
        total = sum(mC[i][s] * wX[s] for s in range(i + 1, n))
        wX[i] = (mC[i][n] - total) / mC[i][i]
    return wX

def spline_interpolation(X, Y):
    n = X.shape[0]
    M = np.zeros([n + 2, n + 2])
    FX = np.zeros(n + 2)
    h = X[1] - X[0]
    
    for i in range(n + 2):
        for j in range(n + 2):
            if i == j == 0 or (i == n + 1 and j == n - 1):
                M[i][j] = -3 / h
            elif i == j == n + 1 or (i == 0 and j == 2):
                M[i][j] = 3 / h
            elif i == j:
                M[i][j] = 4
            elif not ((i == 0 and j == 1) or (i == n + 1 and j == n)) and (j == i + 1 or j == i - 1):
                M[i][j] = 1

    FX[0] = 1
    FX[n + 1] = -1
    for i in range(1, n + 1):
        FX[i] = Y[i - 1]

    return gaussian_elimination(M, FX)

def b_spline_basis(x, h):
    if -2*h <= x <= -h:
        y = (x + 2*h)**3
    elif -h <= x <= 0:
        y = (h**3) + 3*(h**2)*(x + h) + 3*h*((x + h)**2) - 3*((x + h)**3)
    elif 0 <= x <= h:
        y = (h**3) + 3*(h**2)*(h - x) + 3*h*((h - x)**2) - 3*((h - x)**3)
    elif h <= x <= 2*h:
        y = (2*h - x)**3
    else:
        y = 0
    return y * (1 / (h**3))

def spline_function(x, X, K):
    n = X.shape[0]
    h = X[1] - X[0]
    xT = np.zeros(n + 2)
    xT[0] = X[0] - h
    for i in range(1, n + 1):
        xT[i] = X[i - 1]
    xT[n + 1] = X[n - 1] + h

    S = sum(K[i] * b_spline_basis(x - xT[i], h) for i in range(n + 2))
    return S

def linear_approximation(X, Y):
    n = X.shape[0]
    S = [sum(X), sum(X**2), sum(Y), sum(Y * X)]
    
    M = np.array([[n, S[0]], [S[0], S[1]]])
    y = np.array([S[2], S[3]])
    return gaussian_elimination(M, y)

def quadratic_approximation(X, Y):
    n = X.shape[0]
    S = [sum(X), sum(X**2), sum(X**3), sum(X**4), sum(Y), sum(Y * X), sum(Y * (X**2))]
    
    M = np.array([
        [n, S[0], S[1]],
        [S[0], S[1], S[2]],
        [S[1], S[2], S[3]]
    ])
    y = np.array([S[4], S[5], S[6]])
    return gaussian_elimination(M, y)

def polynomial_func(A, x):
    return sum(A[i] * (x**i) for i in range(A.shape[0]))

# Numerical Integration (Rectangle & Simpson methods combined for brevity)
def integrate_rect_poly(A, a, b, n):
    h = (b - a) / n
    return sum(h * polynomial_func(A, a + i * h) for i in range(n))

def integrate_simpson_poly(A, a, b, n):
    h = (b - a) / n
    return sum((h / 3) * (polynomial_func(A, a + i * h) + 4 * polynomial_func(A, a + i * h + h) + polynomial_func(A, a + i * h + 2 * h)) for i in range(0, n, 2))

def integrate_rect_spline(A, a, b, n, X):
    h = (b - a) / n
    return sum(h * spline_function(a + i * h, X, A) for i in range(n))

def integrate_simpson_spline(A, a, b, n, X):
    h = (b - a) / n
    return sum((h / 3) * (spline_function(a + i * h, X, A) + 4 * spline_function(a + i * h + h, X, A) + spline_function(a + i * h + 2 * h, X, A)) for i in range(0, n, 2))

def first_derivative(X, Z):
    Zp = np.zeros(Z.shape[0])
    for i in range(Zp.shape[0]):
        if i == 0:
            Zp[i] = (Z[i + 1] - Z[i]) / (X[i + 1] - X[i])
        elif i == Zp.shape[0] - 1:
            Zp[i] = (Z[i] - Z[i - 1]) / (X[i] - X[i - 1])
        else:
            Zp[i] = (Z[i + 1] - Z[i - 1]) / (X[i + 1] - X[i - 1])
    return Zp

def triangle_area(x1, y1, z1, x2, y2, z2, x3, y3, z3):
    ux = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
    uy = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
    uz = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    return 0.5 * np.sqrt(ux**2 + uy**2 + uz**2)