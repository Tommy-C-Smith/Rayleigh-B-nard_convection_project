#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 00:39:30 2025

@author: tommycursonsmith
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable

def extract_snapshot_number(filename):
    match = re.search(r'snapshots_3D_s(\d+).h5', filename)
    return int(match.group(1)) if match else -1

snapshot_files = sorted(glob.glob('snapshots_3D/snapshots_3D_s*.h5'), key=extract_snapshot_number)
if len(snapshot_files) < 1:
    raise FileNotFoundError("No snapshot files found")

latest_files = snapshot_files[-3:] 

Lx = 8
Ly = 8
x = None
y = None

for file in latest_files:
    with h5py.File(file, 'r') as f:
        b = f['tasks']['buoyancy'][:] 
        vort = f['tasks']['vorticity'][:] 
        times = f['scales']['sim_time'][:]

        if x is None or y is None:
            Nz, Ny, Nx = b.shape[-3:]
            x = np.linspace(0, Lx, Nx)
            y = np.linspace(0, Ly, Ny)

        z_index = 54 
        b_xy = b[-1, :, :, z_index] 
        vort_xy = vort[-1, :, :, z_index] 
        
        extent = (x.min(), x.max(), y.min(), y.max())

    
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        im1 = ax1.imshow(b_xy, extent=extent, origin='lower',
                         cmap='plasma', interpolation='bicubic', aspect='auto')
        cbar1 = fig1.colorbar(im1, ax=ax1)
        cbar1.set_label("Buoyancy", fontsize=15)
        cbar1.ax.tick_params(labelsize=12)
        ax1.set_title(f"Buoyancy field at z = {z_index/63:.2f}, t = {times[-1]:.2f} ", fontsize =18)
        ax1.set_xlabel("x-axis", fontsize=15)
        ax1.set_ylabel("y-axis", fontsize=15)
        ax1.tick_params(labelsize=12)
        plt.tight_layout()
        plt.show()

        
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        im2 = ax2.imshow(vort_xy, extent=extent, origin='lower',
                         cmap='coolwarm', interpolation='bicubic', aspect='auto')
        cbar2 = fig2.colorbar(im2, ax=ax2)
        cbar2.set_label("Vorticity", fontsize=15)
        cbar2.ax.tick_params(labelsize=12)
        ax2.set_title(f"Vorticity at z = {z_index/63:.2f}, t = {times[-1]:.2f}", fontsize =18)
        ax2.set_xlabel("x-axis", fontsize=15)
        ax2.set_ylabel("y-axis", fontsize=15)
        ax2.tick_params(labelsize=12)
        plt.tight_layout()
        plt.show()

