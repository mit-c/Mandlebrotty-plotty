import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
plt.style.use("seaborn-pastel")

num_its = 50
width = 300
height = 300
fig = plt.figure()
ax = plt.axes()


def mandlebrot(num_its, real_range, imag_range, interesting_point, theta):
    out_grid = [[0 for i in range(len(real_range))] for j in range(len(imag_range))]

    for i in range(width):
        for j in range(height):
            x = real_range[i]
            y = imag_range[j]
            xnew, ynew = rotate([x,y],interesting_point,theta)
            c = complex(xnew, ynew)
            new_z = complex(0, 0)
            for k in range(num_its):
                new_z = new_z * new_z + c
                if abs(new_z) > 4:
                    break
                out_grid[j][i] = k / num_its
    return out_grid


def animate(i, zoom_list_real, zoom_list_imag, interesting_point,theta_list):
    print(i)
    ax.clear()
    ax.tick_params(axis="both",
                   which="both",
                   bottom=False,
                   top=False,
                   left=False,
                   labelbottom=False,
                   labelleft=False) # learning what this stuff means (I think a lot is unnecessary).


    out_grid = mandlebrot(num_its=i+20,
                          real_range=zoom_list_real[i],
                          imag_range=zoom_list_imag[i],
                          interesting_point=interesting_point,
                          theta=theta_list[i])
    img = ax.imshow(out_grid, origin='lower')
    return [img]

def rotate(initial_coord, rotate_point, theta):
    s = np.sin(theta)
    c = np.cos(theta)
    xnew = initial_coord[0] - rotate_point[0]
    ynew = initial_coord[1] - rotate_point[1]
    xnew = xnew*c - ynew*s
    ynew = xnew*s + ynew*c
    xnew += rotate_point[0]
    ynew += rotate_point[1]
    return xnew, ynew


def zoom(real_range, imag_range, zoom_factor, interesting_point):
    width = len(real_range)
    height = len(imag_range)
    min_real = real_range[0]
    max_real = real_range[-1]
    min_imag = imag_range[0]
    max_imag = imag_range[-1]

    new_real_range = zoom_factor*(max_real - min_real)
    new_imag_range = zoom_factor*(max_imag - min_imag)
    new_min_real = interesting_point[0] - new_real_range/2
    new_min_imag = interesting_point[1] - new_imag_range/2
    out_real = [new_min_real + new_real_range * i / width for i in range(width)]
    out_imag = [new_min_imag + new_imag_range * j / height for j in range(height)]

    return out_real, out_imag

theta_bool = True
num_frames = 30
zoom_factor = 0.8
interesting_point = [-0.74, 0.36] # [real, complex]
initial_range = [4, 2]
# preprocess zoom ranges
real_range = [interesting_point[0] - initial_range[0]/2 + initial_range[0] * i / width for i in range(width)]   # interesting point - half of range + range add.
imag_range = [interesting_point[1] - initial_range[1]/2 + initial_range[1] * j / height for j in range(height)]
zoom_list_real = [real_range]
zoom_list_imag = [imag_range]


theta_list = [0]
theta_add = 2*np.pi/(6*num_frames)
theta = 0
for i in range(num_frames-1):
        real_range, imag_range = zoom(real_range, imag_range, zoom_factor, interesting_point)
        theta += theta_add
        theta_list.append(theta)
        zoom_list_real.append(real_range)
        zoom_list_imag.append(imag_range)


if not theta_bool:
    theta_list = [0]*num_frames

mandlebrot_ani = FuncAnimation(fig, animate, frames=num_frames, interval=120, blit=True, fargs=(zoom_list_real, zoom_list_imag, interesting_point, theta_list))
mandlebrot_ani.save("mandlebrot_plot.gif", writer="imagemagick")
