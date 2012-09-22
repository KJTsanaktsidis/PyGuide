
from waveguides import PlanarWaveguideInterp
from modesolvers import LossySolver
from numpy import *
import matplotlib
matplotlib.use('QT4Agg')
from matplotlib.pyplot import *
from numpy.linalg import *
from util import *

wg = PlanarWaveguideInterp.get_waveguide('losswg')
s = LossySolver(wg, 1.54e-10)

kxs = s.solve_transcendental()
kxn2 = kxs[1] #work with n=2 mode

n = s.waveguide.cladding_index(s.wavelength)
k = s.waveguide.wavevector_length_core(s.wavelength).real
d = s.waveguide.slab_gap

gamma = lambda kx: sqrt(k ** 2 - kx ** 2 - n.real ** 2 * k ** 2)
def f_to_zero(kx):
    #Do every calculation seperately and nail this loss of precsion down!
    i1 = k**2
    i2 = kx**2
    i3 = n**2
    i4 = i3*i1
    i5 = i1 - i2 - i4
    i6 = sqrt(i5) #gamma
    i7 = kx * d
    i8 = tan(i7)
    i9 = 2*kx
    i10 = i9 * i6
    i11 = i2 - i5
    i12 = i10 / i11
    i13 = i8 - i12
    return i13

dr = 1.5e-7
di = 1.5e-10

vf = vectorize(f_to_zero)
xmin = kxn2.real - dr
xmax = kxn2.real + dr
ymin = kxn2.imag - di
ymax = kxn2.imag + di
x = linspace(xmin, xmax, 500)
y = linspace(ymin, ymax, 500)
X, Y = meshgrid(x, y)
Z = vf(X+Y*1j)
reZ = Z.real
imZ = Z.imag
CN = zeros((500, 500))
Residual = zeros((500, 500))
for n,xv in enumerate(x):
    for j,yv in enumerate(y):
        kx = xv+yv*1j
        matr = s.coeff_matrix(kx)
        nullv = null_vector(matr)
        Residual[n,j] = sum(abs(nullv))
        CN[n,j] = cond(matr)


xticks = linspace(xmin, xmax, 6)
yticks = linspace(ymin, ymax, 6)

figure()
subplot(121)
hold(True)
imshow(reZ, extent=(xmin, xmax, ymin, ymax), origin='lower')
#imshow(reZ, extent=(xmin, xmax, ymin, ymax), origin='upper')
jet()
colorbar()
CSRe = contour(reZ, 1, colors='black', origin='lower', extent=(xmin, xmax, ymin, ymax))
gca().clabel(CSRe, inline=1)
gca().set_aspect((xmax-xmin)/(ymax-ymin))
gca().set_xticks(xticks)
gca().set_yticks(yticks)
title('Real part')
xlabel('Re(kx)')
ylabel('Im(kx)')
hold(False)
subplot(122)
hold(True)
imshow(imZ, extent=(xmin, xmax, ymin, ymax), origin='lower')
jet()
colorbar()
CSIm = contour(imZ, 1, colors='black', inline=1, origin='lower', extent=(xmin, xmax, ymin, ymax))
gca().clabel(CSIm, inline=1)
gca().set_aspect((xmax - xmin) / (ymax - ymin))
gca().set_xticks(xticks)
gca().set_yticks(yticks)
title('Imaginary Part')
xlabel('Re(kx)')
ylabel('Im(kx)')
hold(False)

figure()
title('Condition number')
imshow(CN, extent=(xmin, xmax, ymin, ymax), origin='lower')
jet()
colorbar()
gca().set_aspect((xmax - xmin) / (ymax - ymin))
gca().set_xticks(xticks)
gca().set_yticks(yticks)
xlabel('Re(kx)')
ylabel('Im(kx)')

figure()
title('Residual value')
imshow(Residual, extent=(xmin, xmax, ymin, ymax), origin='lower')
jet()
colorbar()
gca().set_aspect((xmax - xmin) / (ymax - ymin))
gca().set_xticks(xticks)
gca().set_yticks(yticks)
xlabel('Re(kx)')
ylabel('Im(kx)')

show()

#for n in xrange(1, 500):
#    for j in xrange(1,500):
#        no = n - 250
#        jo = j - 250
#        x = kx.real + no*dr
#        y = kx.imag + jo*di
#        fval = f_to_zero(x+y*1j)
#        repart[n,j] = fval.real
#        repart[n,j] = fval.imag

