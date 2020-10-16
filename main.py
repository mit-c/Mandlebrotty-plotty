import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from matplotlib.animation import FuncAnimation
from functools import partial

# TODO: There is something causing rotate to stretch images.
plt.style.use("seaborn-pastel")

fig = plt.figure()
ax = plt.axes()


def inner_loop(real_range, num_its, y, interesting_point, theta):
    # This takes real_range (an iterable) and outputs an iterable for each one.
    # It should therefore just output a colour and that should get mapped to an iterable which I can store.

    x = real_range
    xnew, ynew = rotate((x, y), interesting_point, theta)
    c = complex(xnew, ynew)
    new_z = complex(0, 0)
    for k in range(num_its):
        new_z = new_z * new_z + c
        if abs(new_z) > 4:
            break

    out_colour = k / num_its
    return out_colour


def mandlebrot(num_its, real_range, imag_range, interesting_point, theta):
    height = len(imag_range)
    out_grid = [0 for j in range(len(real_range))]
    pool = Pool(processes=4)  # TODO probably should preprocess rotation matrix.
    for j in range(height):
        just_inner_loop = partial(inner_loop, num_its=num_its, y=imag_range[j], interesting_point=interesting_point,
                                  theta=theta)
        row = pool.map(just_inner_loop, real_range)
        out_grid[j] = row

    pool.close()
    pool.join()
    # The code below normalises the colour will either look good or terrible.
    min_val = min(min(out_grid))
    max_val = max(max(out_grid))
    if max_val - min_val != 0:
        new_out_grid = [[(val - min_val) / (max_val - min_val) for val in out_grid_row] for out_grid_row in out_grid]
        return new_out_grid
    return out_grid


def animate(i, zoom_list_real, zoom_list_imag, interesting_point, theta_list, it_mult):
    print("Frame number:", i + 1)
    ax.clear()
    ax.tick_params(axis="both",
                   which="both",
                   bottom=False,
                   top=False,
                   left=False,
                   labelbottom=False,
                   labelleft=False)  # learning what this stuff means (I think a lot is unnecessary).

    num_its = int(i * it_mult + 50)
    print("number of iterations", num_its)
    out_grid = mandlebrot(num_its=num_its,
                          real_range=zoom_list_real[i],
                          imag_range=zoom_list_imag[i],
                          interesting_point=interesting_point,
                          theta=theta_list[i])

    img = ax.imshow(out_grid,
                    origin='lower')
    return [img]


def rotate(initial_coord, rotate_point, theta):
    s, c = np.sin(theta), np.cos(theta)
    xold = initial_coord[0] - rotate_point[0]
    yold = initial_coord[1] - rotate_point[1]
    xnew = xold * c - yold * s
    ynew = xold * s + yold * c
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

    new_real_range = zoom_factor * (max_real - min_real)
    new_imag_range = zoom_factor * (max_imag - min_imag)
    new_min_real = interesting_point[0] - new_real_range / 2
    new_min_imag = interesting_point[1] - new_imag_range / 2
    out_real = [new_min_real + new_real_range * i / width for i in range(width)]
    out_imag = [new_min_imag + new_imag_range * j / height for j in range(height)]

    return out_real, out_imag


def main():
    width = 300
    height = 300
    num_frames = 120  # total number of frames in gif
    theta_bool = True  # if set to False no rotation will be used -- makes finding interesting point easier.
    zoom_factor = 0.8  # how much zoom per frame e.g. 0.8 means zoom in by 20%
    it_mult = 5  # I think the relationship between zoom factor and it_power is bigger than linear (testing quadratic).
    interesting_point = [-0.77568377, 0.13646737]  # [-0.7285, 0.3583]  # [real, complex]
    print("zoom point: ", interesting_point)
    print("total frames: ", num_frames)
    initial_range = [4, 2]
    # preprocessing lists for zoom.
    real_range = [interesting_point[0] - initial_range[0] / 2 + initial_range[0] * i / width for i in
                  range(width)]  # interesting point - half of range + range addition.
    imag_range = [interesting_point[1] - initial_range[1] / 2 + initial_range[1] * j / height for j in range(height)]
    zoom_list_real = [real_range]
    zoom_list_imag = [imag_range]

    theta_list = [0]
    theta_add = 2 * np.pi / (4 * num_frames)
    theta = 0
    for i in range(num_frames - 1):
        real_range, imag_range = zoom(real_range, imag_range, zoom_factor, interesting_point)
        theta += theta_add
        theta_list.append(theta)
        zoom_list_real.append(real_range)
        zoom_list_imag.append(imag_range)

    if not theta_bool:
        theta_list = [0] * num_frames

    mandlebrot_ani = FuncAnimation(fig, animate, frames=num_frames, interval=120, blit=True,
                                   fargs=(zoom_list_real, zoom_list_imag, interesting_point, theta_list, it_mult))
    mandlebrot_ani.save("mandlebrot_plot.gif", writer="imagemagick")
    '''
    index = -1
    imag_range = zoom_list_imag[index]
    real_range = zoom_list_real[index]
    theta_test = theta_list[index]
    out_grid=mandlebrot(num_its=600, real_range=real_range,imag_range=imag_range,interesting_point=interesting_point,theta=theta_test)
    print(np.sin(theta_test), np.cos(theta_test))
    plt.imshow(out_grid)
    plt.show()
    '''


if __name__ == "__main__":
    main()
