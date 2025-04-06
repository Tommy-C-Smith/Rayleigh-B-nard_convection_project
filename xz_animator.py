#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 00:54:31 2025

@author: tommycursonsmith
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import glob
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable

# === Sort snapshot files numerically ===
def extract_snapshot_number(filename):
    match = re.search(r'snapshots_2D_s(\d+).h5', filename)
    return int(match.group(1)) if match else -1

snapshot_files = sorted(glob.glob('snapshots_2D/snapshots_2D_s*.h5'), key=extract_snapshot_number)
if not snapshot_files:
    raise FileNotFoundError("No snapshot files found.")

# === Domain info ===
Lx, Lz = 8, 1
x = z = None

# === Collect all rotated frames ===
buoyancy_frames = []
vorticity_frames = []
time_stamps = []

for file in snapshot_files[-4:]:  # Last 4 snapshots
    with h5py.File(file, 'r') as f:
        b = f['tasks']['buoyancy'][:]     # (Nt, Nz, Nx)
        vort = f['tasks']['vorticity'][:] # (Nt, Nz, Nx)
        times = f['scales']['sim_time'][:]

        if x is None or z is None:
            Nz, Nx = b.shape[-2:]
            x = np.linspace(0, Lx, Nx)
            z = np.linspace(0, Lz, Nz)

        for i in range(b.shape[0]):
            b_rot = np.rot90(b[i], k=-1)
            vort_rot = np.rot90(vort[i], k=-1)
            buoyancy_frames.append(b_rot)
            vorticity_frames.append(vort_rot)
            time_stamps.append(times[i])

# === Set up figure and axes ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

# --- Initial frame ---
pcm1 = ax1.pcolormesh(z, x, buoyancy_frames[0], shading='auto', cmap='plasma')
ax1.set_xticks(np.linspace(z[0], z[-1], 5))  # z axis (used for horizontal) becomes x
ax1.set_xticklabels(np.linspace(0, Lx, 5))   # relabel it as if it's x
ax1.set_yticks(np.linspace(x[0], x[-1], 5))  # x axis (used for vertical) becomes z
ax1.set_yticklabels(np.linspace(0, Lz, 5))
divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="2%", pad=0.07)
cbar1 = fig.colorbar(pcm1, cax=cax1)
cbar1.set_label("Buoyancy", fontsize=15)
cbar1.ax.tick_params(labelsize=12)
ax1.set_title(f"Buoyancy at t = {time_stamps[0]:.2f}")
ax1.set_ylabel("z-axis", fontsize=15)
ax1.tick_params(labelsize=12)

pcm2 = ax2.pcolormesh(z, x, vorticity_frames[0], shading='auto',
                      cmap='coolwarm', vmin=-10, vmax=10)
ax2.set_xticks(np.linspace(z[0], z[-1], 5))
ax2.set_xticklabels(np.linspace(0, Lx, 5))
ax2.set_yticks(np.linspace(x[0], x[-1], 5))
ax2.set_yticklabels(np.linspace(0, Lz, 5))
divider2 = make_axes_locatable(ax2)
cax2 = divider2.append_axes("right", size="2%", pad=0.07)
cbar2 = fig.colorbar(pcm2, cax=cax2)
cbar2.set_label("Vorticity", fontsize=15)
cbar2.ax.tick_params(labelsize=12)
ax2.set_title(f"Vorticity at t = {time_stamps[0]:.2f}")
ax2.set_xlabel("x-axis", fontsize=15)
ax2.set_ylabel("z-axis", fontsize=15)
ax2.tick_params(labelsize=12)



# === Animation update function ===
def update(frame_idx):
    pcm1.set_array(buoyancy_frames[frame_idx].ravel())
    pcm2.set_array(vorticity_frames[frame_idx].ravel())
    ax1.set_title(f"Buoyancy at t = {time_stamps[frame_idx]:.2f}")
    ax2.set_title(f"Vorticity at t = {time_stamps[frame_idx]:.2f}")
    return pcm1, pcm2, 

# === Create and save animation ===
ani = animation.FuncAnimation(fig, update, frames=len(buoyancy_frames), interval=50, blit=False)

print("Saving GIF...")
ani.save("xz_convection.gif", writer='pillow', fps=20)
print("Saved as rb_convection.gif")

plt.tight_layout()
plt.show()




