import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# USER FLAG
# =========================================================
ROTATED = True # False for Non-Rotated Ellipses

# =========================================================
# GENERATE ELLIPSE
# =========================================================
def generate_ellipse(a, b, theta=0, n=80):
    t = np.linspace(0, 2*np.pi, n)
    x = a * np.cos(t)
    y = b * np.sin(t)

    if theta != 0:
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        pts = R @ np.vstack((x, y))
        x, y = pts[0], pts[1]

    return np.column_stack((x, y))


# =========================================================
# DATA
# =========================================================
if ROTATED:
    pts1 = generate_ellipse(5, 3, theta=np.deg2rad(30))
    pts2 = generate_ellipse(8, 2, theta=np.deg2rad(60))
else:
    pts1 = generate_ellipse(5, 3)
    pts2 = generate_ellipse(8, 2)

points = np.vstack((pts1, pts2))


# =========================================================
# FIT 3-POINT MODEL
# =========================================================
def fit_ellipse(pts):
    M = np.array([[x**2, x*y, y**2] for x, y in pts])
    b = np.ones(3)
    return np.linalg.solve(M, b)


# =========================================================
# SEGREGATION
# =========================================================
def segregate(points, theta):
    A, B, C = theta
    D = []

    for x, y in points:
        val = A*x**2 + B*x*y + C*y**2
        if abs(val - 1) < 5e-2:   # relaxed tolerance
            D.append([x, y])

    return np.array(D)


# =========================================================
# LEAST SQUARES REFINEMENT
# =========================================================
def fit_ls(points):
    M = np.array([[x**2, x*y, y**2] for x, y in points])
    b = np.ones(len(points))
    return np.linalg.lstsq(M, b, rcond=None)[0]


# =========================================================
# PARAMETER EXTRACTION
# =========================================================
def compute_parameters(theta):
    A, B, C = theta

    Q = np.array([[A, B/2],
                  [B/2, C]])

    eigvals, eigvecs = np.linalg.eig(Q)

    # sort eigenvalues (important)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    a = np.sqrt(1 / eigvals[0])
    b = np.sqrt(1 / eigvals[1])
    e = np.sqrt(1 - (b**2 / a**2))

    return a, b, e, eigvecs


# =========================================================
# MAIN PIPELINE
# =========================================================
theta1 = fit_ellipse(pts1[:3])
theta2 = fit_ellipse(pts2[:3])

D1 = segregate(points, theta1)
D2 = segregate(points, theta2)

# ✔ refinement step (VERY IMPORTANT)
theta1 = fit_ls(D1)
theta2 = fit_ls(D2)

a1, b1, e1, vec1 = compute_parameters(theta1)
a2, b2, e2, vec2 = compute_parameters(theta2)

print("Ellipse 1: a =", a1, "b =", b1, "e =", e1)
print("Ellipse 2: a =", a2, "b =", b2, "e =", e2)


# =========================================================
# PLOTTING
# =========================================================
plt.figure(figsize=(6,6))

plt.scatter(points[:,0], points[:,1], color='gray', label='All Points')
plt.scatter(D1[:,0], D1[:,1], color='blue', label='Ellipse 1')
plt.scatter(D2[:,0], D2[:,1], color='red', label='Ellipse 2')


def plot_ellipse(theta, color):
    A, B, C = theta

    Q = np.array([[A, B/2],
                  [B/2, C]])

    eigvals, eigvecs = np.linalg.eig(Q)

    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    a = np.sqrt(1 / eigvals[0])
    b = np.sqrt(1 / eigvals[1])

    t = np.linspace(0, 2*np.pi, 200)
    ellipse = np.array([a*np.cos(t), b*np.sin(t)])

    ellipse_rot = eigvecs @ ellipse

    plt.plot(ellipse_rot[0], ellipse_rot[1], color=color, linewidth=2)


plot_ellipse(theta1, 'blue')
plot_ellipse(theta2, 'red')

plt.title("Ellipse Segregation (Fixed)")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.legend()
plt.grid()

plt.show()
