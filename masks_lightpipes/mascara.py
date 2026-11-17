from diffractio import sp, nm, plt, np, mm, degrees, um
from diffractio.scalar_fields_XY import Scalar_field_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XY import Scalar_mask_XY
import numpy as np

N = 0
K = 3

x0 = np.linspace(-1*mm, 1*mm, 512)
y0 = np.linspace(-1*mm, 1*mm, 512)
wavelength = 0.6238*um

u = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)

ID1 = plt.figure(figsize=(8, 8))
plt.suptitle("amplitude", fontsize=20)
ID2 = plt.figure(figsize=(8, 8))
plt.suptitle("phase", fontsize=20)

for n in range(N + 1):
    for k in range(K + 1):

        u.laguerre_beam(A=1,
                        n=n,
                        l=k,
                        r0=(0*um, 0*um),
                        w0=250*um,
                        z0=0,
                        z=0.01*um)

        title = "(%d,%d)" % (n, k)
        plt.figure(ID1.number)
        plt.subplot(N + 1, K + 1, (N + 1) * (k) + n + 1)
        plt.axis('off')
        plt.title(title, fontsize=14)
        h1 = plt.imshow(np.abs(u.u)**2)
        h1.set_cmap("gist_heat")

        plt.figure(ID2.number)
        plt.subplot(N + 1, K + 1, (N + 1) * (k) + n + 1)
        plt.axis('off')
        plt.title(title, fontsize=14)
        h2 = plt.imshow(np.angle(u.u))
        h2.set_cmap("gray")

plt.figure(ID1.number)
plt.figure(ID2.number)