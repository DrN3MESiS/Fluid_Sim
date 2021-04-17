"""
Based on the Jos Stam paper https://www.researchgate.net/publication/2560062_Real-Time_Fluid_Dynamics_for_Games
and the mike ash vulgarization https://mikeash.com/pyblog/fluid-simulation-for-dummies.html

https://github.com/Guilouf/python_realtime_fluidsim
"""
import numpy as np
import math
import json
import os
import argparse


class Fluid:

    def __init__(self):
        self.rotx = 1
        self.roty = 1
        self.cntx = 1
        self.cnty = -1

        self.size = 60  # map size
        self.dt = 0.2  # time interval
        self.iter = 2  # linear equation solving iteration number

        self.diff = 0.0000  # Diffusion
        self.visc = 0.0000  # viscosity

        self.s = np.full((self.size, self.size), 0,
                         dtype=float)        # Previous density
        self.density = np.full((self.size, self.size), 0,
                               dtype=float)  # Current density

        # array of 2d vectors, [x, y]
        self.velo = np.full((self.size, self.size, 2), 0, dtype=float)
        self.velo0 = np.full((self.size, self.size, 2), 0, dtype=float)

    def step(self):
        self.diffuse(self.velo0, self.velo, self.visc)

        # x0, y0, x, y
        self.project(self.velo0[:, :, 0], self.velo0[:, :, 1],
                     self.velo[:, :, 0], self.velo[:, :, 1])

        self.advect(self.velo[:, :, 0], self.velo0[:, :, 0], self.velo0)
        self.advect(self.velo[:, :, 1], self.velo0[:, :, 1], self.velo0)

        self.project(self.velo[:, :, 0], self.velo[:, :, 1],
                     self.velo0[:, :, 0], self.velo0[:, :, 1])

        self.diffuse(self.s, self.density, self.diff)

        self.advect(self.density, self.s, self.velo)

    def lin_solve(self, x, x0, a, c):
        """Implementation of the Gauss-Seidel relaxation"""
        c_recip = 1 / c

        for iteration in range(0, self.iter):
            # Calculates the interactions with the 4 closest neighbors
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (x[2:, 1:-1] +
                                                   x[:-2, 1:-1] + x[1:-1, 2:] + x[1:-1, :-2])) * c_recip

            self.set_boundaries(x)

    def set_boundaries(self, table):
        """
        Boundaries handling
        :return:
        """

        if len(table.shape) > 2:  # 3d velocity vector array
            # Simulating the bouncing effect of the velocity array
            # vertical, invert if y vector
            table[:, 0, 1] = - table[:, 0, 1]
            table[:, self.size - 1, 1] = - table[:, self.size - 1, 1]

            # horizontal, invert if x vector
            table[0, :, 0] = - table[0, :, 0]
            table[self.size - 1, :, 0] = - table[self.size - 1, :, 0]

        table[0, 0] = 0.5 * (table[1, 0] + table[0, 1])
        table[0, self.size - 1] = 0.5 * \
            (table[1, self.size - 1] + table[0, self.size - 2])
        table[self.size - 1, 0] = 0.5 * \
            (table[self.size - 2, 0] + table[self.size - 1, 1])
        table[self.size - 1, self.size - 1] = 0.5 * table[self.size - 2, self.size - 1] + \
            table[self.size - 1, self.size - 2]

    def diffuse(self, x, x0, diff):
        if diff != 0:
            a = self.dt * diff * (self.size - 2) * (self.size - 2)
            self.lin_solve(x, x0, a, 1 + 6 * a)
        else:  # equivalent to lin_solve with a = 0
            x[:, :] = x0[:, :]

    def project(self, velo_x, velo_y, p, div):
        # numpy equivalent to this in a for loop:
        # div[i, j] = -0.5 * (velo_x[i + 1, j] - velo_x[i - 1, j] + velo_y[i, j + 1] - velo_y[i, j - 1]) / self.size
        div[1:-1, 1:-1] = -0.5 * (
            velo_x[2:, 1:-1] - velo_x[:-2, 1:-1] +
            velo_y[1:-1, 2:] - velo_y[1:-1, :-2]) / self.size
        p[:, :] = 0

        self.set_boundaries(div)
        self.set_boundaries(p)
        self.lin_solve(p, div, 1, 6)

        velo_x[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) * self.size
        velo_y[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) * self.size

        self.set_boundaries(self.velo)

    def advect(self, d, d0, velocity):
        dtx = self.dt * (self.size - 2)
        dty = self.dt * (self.size - 2)

        for j in range(1, self.size - 1):
            for i in range(1, self.size - 1):
                tmp1 = dtx * velocity[i, j, 0]
                tmp2 = dty * velocity[i, j, 1]
                x = i - tmp1
                y = j - tmp2

                if x < 0.5:
                    x = 0.5
                if x > (self.size - 1) - 0.5:
                    x = (self.size - 1) - 0.5
                i0 = math.floor(x)
                i1 = i0 + 1.0

                if y < 0.5:
                    y = 0.5
                if y > (self.size - 1) - 0.5:
                    y = (self.size - 1) - 0.5
                j0 = math.floor(y)
                j1 = j0 + 1.0

                s1 = x - i0
                s0 = 1.0 - s1
                t1 = y - j0
                t0 = 1.0 - t1

                i0i = int(i0)
                i1i = int(i1)
                j0i = int(j0)
                j1i = int(j1)

                try:
                    d[i, j] = s0 * (t0 * d0[i0i, j0i] + t1 * d0[i0i, j1i]) + \
                        s1 * (t0 * d0[i1i, j0i] + t1 * d0[i1i, j1i])
                except IndexError:
                    # tmp = str("inline: i0: %d, j0: %d, i1: %d, j1: %d" % (i0, j0, i1, j1))
                    # print("tmp: %s\ntmp1: %s" %(tmp, tmp1))
                    raise IndexError
        self.set_boundaries(d)

    def turn(self):
        self.cntx += 1
        self.cnty += 1
        if self.cntx == 3:
            self.cntx = -1
            self.rotx = 0
        elif self.cntx == 0:
            self.rotx = self.roty * -1
        if self.cnty == 3:
            self.cnty = -1
            self.roty = 0
        elif self.cnty == 0:
            self.roty = self.rotx
        return self.rotx, self.roty


def _LoadDataFromJSON(filename: str) -> dict:
    configFile = open(filename, "r")

    data = json.loads(configFile.read())
    configFile.close()
    data["mode"] = "json"

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Runs DrN3MESiS (Alan Maldonado) Implementation of Fluid Simulation.")
    parser.add_argument(
        "configFilename", help="Filename of the Configuration File")

    args = parser.parse_args()
    if not os.path.isfile(args.configFilename):
        print("[NOT_FOUND] File was not found or is not accessible")
        exit()

    config = _LoadDataFromJSON(args.configFilename)

    class Staging:
        INTERVAL_BOOL = False

        def update_im(self, i):
            # print(f"Rendered Frame #{i}")
            ANIM_SPIN_SPEED = 10
            # DENSITIES
            denSrc = config.get('densitySources', [])
            for src in denSrc:
                x0 = src.get('centerX')
                y0 = src.get('centerY')
                srcSize = src.get("size")
                pwo = src.get('power')

                if x0 < srcSize or x0 > (inst.size-(srcSize+1)):
                    continue
                if y0 < srcSize or y0 > (inst.size-(srcSize+1)):
                    continue

                cx0 = x0-srcSize
                cy0 = y0-srcSize
                cx1 = x0+srcSize
                cy1 = y0+srcSize

                inst.density[cx0:cx1, cy0:cy1] += pwo

            # VELOCITIES
            velSrc = config.get('velocitySources', [])
            for src in velSrc:
                x0 = src.get('X')
                y0 = src.get('Y')
                vec = src.get('vec')
                animType = src.get('animation', "regular")

                if animType == "spin":
                    angle = i * math.pi/180 * ANIM_SPIN_SPEED
                    rotMat = np.array(
                        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
                    P = [vec[0], vec[1]]

                    R = np.matmul(P, rotMat)
                    inst.velo[x0, y0] = R
                elif animType == "reflection":
                    rotMat = np.array(
                        [[-1, 0], [0, -1]])
                    P = [vec[0], vec[1]]

                    if i % 20 == 0:
                        self.INTERVAL_BOOL = not self.INTERVAL_BOOL

                    R = np.matmul(P, rotMat)
                    if self.INTERVAL_BOOL:
                        inst.velo[x0, y0] = R
                    else:
                        inst.velo[x0, y0] = P
                else:
                    inst.velo[x0, y0] = [vec[0], vec[1]]

            # OBJECTS
            objs = config.get('objects', [])
            for obj in objs:
                x0 = obj.get("centerX", 2)
                y0 = obj.get("centerY", 2)
                objType = obj.get("type")
                objSize = obj.get("size")
                # Maximum Object Size

                if objType == "box":
                    if objSize > 5:
                        objSize = 5

                    if objSize > 2:
                        objSize = objSize - 2

                    if x0 < objSize or x0 > (inst.size-(objSize+1)):
                        continue
                    if y0 < objSize or y0 > (inst.size-(objSize+1)):
                        continue

                    cx0 = x0-objSize
                    cy0 = y0-objSize
                    cx1 = x0+objSize
                    cy1 = y0+objSize

                    inst.density[cx0:cx1, cy0:cy1] = 0

                elif objType == "triangle":
                    if x0 < 3 or x0 > (inst.size-4):
                        continue
                    if y0 < 3 or y0 > (inst.size-4):
                        continue

                    inst.density[x0, y0] = 0
                    inst.density[x0, y0+1] = 0

                    inst.density[x0+1, y0] = 0
                    inst.density[x0, y0-1] = 0
                    inst.density[x0-1, y0] = 0
                    inst.density[x0, y0+1] = 0

                    inst.density[x0+1, y0+1] = 0
                    inst.density[x0+1, y0-1] = 0
                    inst.density[x0-1, y0-1] = 0
                    inst.density[x0-1, y0+1] = 0

                    inst.density[x0-2, y0-1] = 0
                    inst.density[x0+2, y0-1] = 0

                    inst.density[x0, y0+2] = 0

            inst.step()
            im.set_array(inst.density)
            q.set_UVC(inst.velo[:, :, 1], inst.velo[:, :, 0])
            im.autoscale()

    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib import animation

        inst = Fluid()
        stage = Staging()

        fig = plt.figure()

        # plot density

        setup = config.get("setup")
        scheme = setup.get("colorScheme")
        schemes = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r',
                   'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r',
                   'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r'
                   ]

        if not schemes.__contains__(scheme):
            scheme = None
        cmap = plt.get_cmap(scheme if scheme != "" else None)
        im = plt.imshow(inst.density, vmax=100, cmap=cmap,
                        interpolation='bilinear')

        # plot vector field
        q = plt.quiver(inst.velo[:, :, 1],
                       inst.velo[:, :, 0], scale=10, angles='xy')
        anim = animation.FuncAnimation(fig, stage.update_im, interval=0)
        anim.save("movie_"+args.configFilename.replace(".json", "") +
                  ".mp4", fps=60, extra_args=['-vcodec', 'libx264'])
        plt.show()

    except ImportError:
        import imageio

        frames = 30

        flu = Fluid()

        video = np.full((frames, flu.size, flu.size), 0, dtype=float)

        for step in range(0, frames):
            flu.density[4:7, 4:7] += 100  # add density into a 3*3 square
            flu.velo[5, 5] += [1, 2]

            flu.step()
            video[step] = flu.density

        imageio.mimsave('./video.gif', video.astype('uint8'))
