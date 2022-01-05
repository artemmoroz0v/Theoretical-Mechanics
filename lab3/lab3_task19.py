import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import sympy as sp
import math

Steps = 1000

def formY(y, t, fV, fOm):
    y1,y2,y3,y4 = y
    dydt = [y3,y4,fV(y1,y2,y3,y4),fOm(y1,y2,y3,y4)]
    return dydt

# defining parameters
alpha = math.pi / 6
M = 1
m = 0.1
R = 0.3
c = 20
l0 = 0.2
g = 9.81

# defining t as a symbol (it will be the independent variable)
t = sp.Symbol('t')

# defining phi, ksi, Vphi=dphi/dt and Vpsi=dpsi/dt as functions of 't'
phi = sp.Function('phi')(t)
psi = sp.Function('psi')(t)
Vphi = sp.Function('Vphi')(t)
Vpsi = sp.Function('Vpsi')(t)
l = 2 * R * sp.cos(phi)  # длина пружины

#1 defining the kinetic energy
TT1 = M * R**2 * Vphi**2 / 2
V1 = 2*Vpsi * R
V2 = Vphi * R * sp.sin(2 * psi)
Vr2 = V1**2 + V2**2
TT2 = m * Vr2 / 2
TT = TT1+TT2

# 2 defining the potential energy
Pi1 = 2 * R * m * g * sp.sin(psi)**2
Pi2 = (c * (l - l0)**2) / 2
Pi = Pi1+Pi2

# 3 Not potential force
M = alpha * phi**2;

# Lagrange function
L = TT-Pi

# equations
ur1 = sp.diff(sp.diff(L,Vphi),t)-sp.diff(L,phi) - M
ur2 = sp.diff(sp.diff(L,Vpsi),t)-sp.diff(L,psi)

# isolating second derivatives(dV/dt and dom/dt) using Kramer's method
a11 = ur1.coeff(sp.diff(Vphi,t),1)
a12 = ur1.coeff(sp.diff(Vpsi,t),1)
a21 = ur2.coeff(sp.diff(Vphi,t),1)
a22 = ur2.coeff(sp.diff(Vpsi,t),1)
b1 = -(ur1.coeff(sp.diff(Vphi,t),0)).coeff(sp.diff(Vpsi,t),0).subs([(sp.diff(phi,t),Vphi), (sp.diff(psi,t), Vpsi)])
b2 = -(ur2.coeff(sp.diff(Vphi,t),0)).coeff(sp.diff(Vpsi,t),0).subs([(sp.diff(phi,t),Vphi), (sp.diff(psi,t), Vpsi)])

detA = a11*a22-a12*a21
detA1 = b1*a22-b2*a21
detA2 = a11*b2-b1*a21

dVdt = detA1/detA
domdt = detA2/detA

countOfFrames = 2000

# Constructing the system of differential equations
T = np.linspace(0, 25, countOfFrames)
fVphi = sp.lambdify([phi,psi,Vphi,Vpsi], dVdt, "numpy")
fVpsi = sp.lambdify([phi,psi,Vphi,Vpsi], domdt, "numpy")
y0 = [0, np.pi/6, -0.5, 0]
sol = odeint(formY, y0, T, args = (fVphi, fVpsi))

t = sp.Symbol('t')
phi = sol[:,0]
psi = sol[:,1]
Vphi = sol[:,2]
Vpsi = sol[:,3]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set(xlim=[-8, 8], ylim=[-8, 8], zlim=[0, 8])
y = 3.5 * np.cos(psi)
z = 3.5 * np.sin(psi)
Z_PointCentral = 8.5
Y_PointCentral = 0
X_PointCentral = 0
PointCentral = ax.plot(X_PointCentral, Y_PointCentral, Z_PointCentral, color='blue', marker='o', markeredgewidth=1)[0]

Z_PointM = 5
X_PointM = 0
Y_PointM = 0
Circle_cir = ax.plot(y * np.cos(phi), y * np.sin(phi), z + 5, linewidth=5, color='red', alpha=.2)[0]
PointM = ax.plot(X_PointM, Y_PointM, Z_PointM, color='black', marker='o', markeredgewidth=4)[0]


def get_spring_line(coils, diameter, start, end):
    x = np.linspace(start[0], end[0], coils * 2)
    y = np.linspace(start[1], end[1], coils * 2)
    z = np.linspace(start[2], end[2], coils * 2)
    for i in range(1, len(z) - 1):
        z[i] = z[i] + diameter * 1 * (-1) ** i
    return np.array([x, y, z], dtype=object)


ax.plot([0, 0], [0, 0], [0, 10.80], linewidth=2, color='black', alpha=.8) #stick
ax.plot([-2.25, -1.25], [-0.25, 0.25], [0.3, 0.3], linewidth=2, color='black', alpha=.8)
ax.plot([-1.25, -1.25], [0.23, 0.25], [-0.3, 0.3], linewidth=2, color='black', alpha=.8)
ax.plot([-1.25, 0.5], [0.25, 1.25], [-0.4, -0.4], linewidth=2, color='black', alpha=.8)
ax.plot([0.5, 0.5], [1.23, 1.25], [-0.3, 0.3], linewidth=2, color='black', alpha=.8)
ax.plot([0.5, 1.40], [1.25, 1.80], [0.3, 0.3], linewidth=2, color='black', alpha=.8)
ax.plot([-0.25, -0.25], [0, 0], [9.75, 10.5], linewidth=1.5, color='black', alpha=.8)

ax.plot([0.25, 0.25], [0.15, 0.15], [9.77, 10.52], linewidth=1.5, color='black', alpha=.8) #right
ax.plot([0.25, 0.5], [0.15, 0.3], [10.52, 10.35], linewidth=1.5, color='black', alpha=.8)
ax.plot([0.25, 0.5], [0.15, 0.3], [10.35, 10.17], linewidth=1.5, color='black', alpha=.8)
ax.plot([0.25, 0.5], [0.15, 0.3], [10.17, 10], linewidth=1.5, color='black', alpha=.8)
ax.plot([0.25, 0.5], [0.15, 0.3], [10, 9.83], linewidth=1.5, color='black', alpha=.8)

ax.plot([-0.25, -0.5], [0, -0.15], [10.5, 10.33], linewidth=1.5, color='black', alpha=.8) #left
ax.plot([-0.25, -0.5], [0, -0.15], [10.33, 10.16], linewidth=1.5, color='black', alpha=.8)
ax.plot([-0.25, -0.5], [0, -0.15], [10.16, 9.99], linewidth=1.5, color='black', alpha=.8)
ax.plot([-0.25, -0.5], [0, -0.15], [9.99, 9.82], linewidth=1.5, color='black', alpha=.8)

#spring
spring_xyz = get_spring_line(30, 0.1, [0, 0, 8.5], [1, 3, 2])
spring = ax.plot(spring_xyz[0], spring_xyz[1], spring_xyz[2], linewidth=2, color='black')[0]

def Kino(i):
    Circle_cir.set_data_3d(y * np.cos(psi[i]), y * np.sin(psi[i]), z + 5)
    curz = (Z_PointM + 3.5 * np.cos(psi[i]))
    if curz >= 5:
        delta = curz - 5
        curz = curz - 2 * delta
    PointM.set_data_3d((X_PointM + 3.5 * np.sin(psi[i])) * np.cos(psi[i]),
                       (Y_PointM + 3.5 * np.sin(psi[i])) * np.sin(psi[i]), curz)
    newx = (X_PointM + 3.5 * np.sin(psi[i])) * np.cos(psi[i])
    newy = (Y_PointM + 3.5 * np.sin(psi[i])) * np.sin(psi[i])
    newz = curz
    spring_xyz = get_spring_line(30, 0.2, [0, 0, 8.5], [newx, newy, newz])
    spring.set_data_3d(spring_xyz[0], spring_xyz[1], spring_xyz[2])
    return [Circle_cir, PointM, spring]

anima = FuncAnimation(fig, Kino, frames=500, interval=80)
plt.show()
