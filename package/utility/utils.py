import os
import pyautogui
import sys
import time
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import tkinter as tk
import xml.etree.ElementTree as ET
from scipy.spatial.distance import cdist
from PIL import Image
from IPython.display import Markdown,display
from xml.dom import minidom
from PyQt5.QtWidgets import (
    QApplication, QWidget, QSlider, QLabel, QGridLayout, QScrollArea, QVBoxLayout, QFrame
)
from PyQt5.QtCore import Qt

"""
    sys.path.append('../../package/kinematics_helper/') # for 'transforms'
"""
from transforms import t2p

def trim_scale(x,th):
    """
        Trim scale
    """
    x         = np.copy(x)
    x_abs_max = np.abs(x).max()
    if x_abs_max > th:
        x = x*th/x_abs_max
    return x

def compute_view_params(
        camera_pos,
        target_pos,
        up_vector = np.array([0,0,1]),
    ):
    """Compute azimuth, distance, elevation, and lookat for a viewer given camera pose in 3D space.

    Args:
        camera_pos (np.ndarray): 3D array of camera position.
        target_pos (np.ndarray): 3D array of target position.
        up_vector (np.ndarray): 3D array of up vector.

    Returns:
        tuple: Tuple containing azimuth, distance, elevation, and lookat values.
    """
    # Compute camera-to-target vector and distance
    cam_to_target = target_pos - camera_pos
    distance = np.linalg.norm(cam_to_target)

    # Compute azimuth and elevation
    azimuth = np.arctan2(cam_to_target[1], cam_to_target[0])
    azimuth = np.rad2deg(azimuth) # [deg]
    elevation = np.arcsin(cam_to_target[2] / distance)
    elevation = np.rad2deg(elevation) # [deg]

    # Compute lookat point
    lookat = target_pos

    # Compute camera orientation matrix
    zaxis = cam_to_target / distance
    xaxis = np.cross(up_vector, zaxis)
    yaxis = np.cross(zaxis, xaxis)
    cam_orient = np.array([xaxis, yaxis, zaxis])

    # Return computed values
    return azimuth, distance, elevation, lookat

def get_idxs(list_query,list_domain):
    """ 
        Get corresponding indices of either two lists or ndarrays
    """
    if isinstance(list_query,list) and isinstance(list_domain,list):
        idxs = [list_query.index(item) for item in list_domain if item in list_query]
    else:
        print("[get_idxs] inputs should be 'List's.")
    return idxs

def get_idxs_contain(list_query,list_substring):
    """ 
        Get corresponding indices of either two lists 
    """
    idxs = [i for i, s in enumerate(list_query) if any(sub in s for sub in list_substring)]
    return idxs

def get_colors(n_color=10,cmap_name='gist_rainbow',alpha=1.0):
    """ 
        Get diverse colors
    """
    colors = [plt.get_cmap(cmap_name)(idx) for idx in np.linspace(0,1,n_color)]
    for idx in range(n_color):
        color = colors[idx]
        colors[idx] = color
    return colors

def sample_xyzs(n_sample,x_range=[0,1],y_range=[0,1],z_range=[0,1],min_dist=0.1,xy_margin=0.0):
    """
        Sample a point in three dimensional space with the minimum distance between points
    """
    xyzs = np.zeros((n_sample,3))
    for p_idx in range(n_sample):
        while True:
            x_rand = np.random.uniform(low=x_range[0]+xy_margin,high=x_range[1]-xy_margin)
            y_rand = np.random.uniform(low=y_range[0]+xy_margin,high=y_range[1]-xy_margin)
            z_rand = np.random.uniform(low=z_range[0],high=z_range[1])
            xyz = np.array([x_rand,y_rand,z_rand])
            if p_idx == 0: break
            devc = cdist(xyz.reshape((-1,3)),xyzs[:p_idx,:].reshape((-1,3)),'euclidean')
            if devc.min() > min_dist: break # minimum distance between objects
        xyzs[p_idx,:] = xyz
    return xyzs

def save_png(img,png_path,verbose=False):
    """ 
        Save image
    """
    directory = os.path.dirname(png_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        if verbose:
            print ("[%s] generated."%(directory))
    # Save to png
    plt.imsave(png_path,img)
    if verbose:
        print ("[%s] saved."%(png_path))
        
class MultiSliderClass(object):
    """
        GUI with multiple sliders
    """
    def __init__(
            self,
            n_slider      = 10,
            title         = 'Multiple Sliders',
            window_width  = 500,
            window_height = None,
            x_offset      = 500,
            y_offset      = 100,
            slider_width  = 400,
            label_texts   = None,
            slider_mins   = None,
            slider_maxs   = None,
            slider_vals   = None,
            resolution    = None,
            resolutions   = None,
            fontsize      = 10,
            verbose       = True
        ):
        """
            Initialze multiple sliders
        """
        self.n_slider      = n_slider
        self.title         = title
        
        self.window_width  = window_width
        if window_height is None:
            self.window_height = self.n_slider*40
        else:
            self.window_height = window_height
        self.x_offset      = x_offset
        self.y_offset      = y_offset
        self.slider_width  = slider_width
        self.resolution    = resolution
        self.resolutions   = resolutions
        self.fontsize      = fontsize
        self.verbose       = verbose
        
        # Slider values
        self.slider_values = np.zeros(self.n_slider)
        
        # Initial/default slider settings
        self.label_texts   = label_texts
        self.slider_mins   = slider_mins
        self.slider_maxs   = slider_maxs
        self.slider_vals   = slider_vals
        
        # Create main window
        self.gui = tk.Tk()
        
        self.gui.title("%s"%(self.title))
        self.gui.geometry(
            "%dx%d+%d+%d"%
            (self.window_width,self.window_height,self.x_offset,self.y_offset))
        
        # Create vertical scrollbar
        self.scrollbar = tk.Scrollbar(self.gui,orient=tk.VERTICAL)
        
        # Create a Canvas widget with the scrollbar attached
        self.canvas = tk.Canvas(self.gui,yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure the scrollbar to control the canvas
        self.scrollbar.config(command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create a frame inside the canvas to hold the sliders
        self.sliders_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0,0),window=self.sliders_frame,anchor=tk.NW)
        
        # Create sliders
        self.sliders = self.create_sliders()
        
        # Update the canvas scroll region when the sliders_frame changes size
        self.sliders_frame.bind("<Configure>",self.cb_scroll)

        # You may want to do this in the main script
        for _ in range(100): self.update() # to avoid GIL-related error 
        
    def cb_scroll(self,event):    
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
    def cb_slider(self,slider_idx,slider_value):
        """
            Slider callback function
        """
        self.slider_values[slider_idx] = slider_value # append
        if self.verbose:
            print ("slider_idx:[%d] slider_value:[%.1f]"%(slider_idx,slider_value))
        
    def create_sliders(self):
        """
            Create sliders
        """
        sliders = []
        for s_idx in range(self.n_slider):
            # Create label
            if self.label_texts is None:
                label_text = "Slider %02d "%(s_idx)
            else:
                label_text = "[%d/%d] %s"%(s_idx,self.n_slider,self.label_texts[s_idx])
            slider_label = tk.Label(self.sliders_frame, text=label_text,font=("Helvetica",self.fontsize))
            slider_label.grid(row=s_idx,column=0,padx=0,pady=0)
            
            # Create slider
            if self.slider_mins is None: slider_min = 0
            else: slider_min = self.slider_mins[s_idx]
            if self.slider_maxs is None: slider_max = 100
            else: slider_max = self.slider_maxs[s_idx]
            if self.slider_vals is None: slider_val = 50
            else: slider_val = self.slider_vals[s_idx]

            # Resolution
            if self.resolution is None: # if none, divide the range with 100
                resolution = (slider_max-slider_min)/100
            else:
                resolution = self.resolution 
            if self.resolutions is not None:
                resolution = self.resolutions[s_idx]

            slider = tk.Scale(
                self.sliders_frame,
                from_      = slider_min,
                to         = slider_max,
                orient     = tk.HORIZONTAL,
                command    = lambda value,idx=s_idx:self.cb_slider(idx,float(value)),
                resolution = resolution,
                length     = self.slider_width,
                font       = ("Helvetica",self.fontsize),
            )
            slider.grid(row=s_idx,column=1,padx=0,pady=0,sticky=tk.W)
            slider.set(slider_val)
            sliders.append(slider)
            
        return sliders
    
    def update(self):
        if self.is_window_exists():
            self.gui.update()
        
    def run(self):
        self.gui.mainloop()
        
    def is_window_exists(self):
        try:
            return self.gui.winfo_exists()
        except tk.TclError:
            return False
        
    def get_slider_values(self):
        return self.slider_values
    
    def set_slider_values(self,slider_values):
        self.slider_values = slider_values
        for slider,slider_value in zip(self.sliders,self.slider_values):
            slider.set(slider_value)

    def set_slider_value(self,slider_idx,slider_value):
        self.slider_values[slider_idx] = slider_value
        slider = self.sliders[slider_idx]
        slider.set(slider_value)
    
    def close(self):
        if self.is_window_exists():
            # some loop
            for _ in range(100): self.update() # to avoid GIL-related error 
            # Close 
            self.gui.destroy()
            self.gui.quit()
            self.gui.update()                
            

class MultiSliderWidgetClass(QWidget):
    def __init__(
            self,
            n_sliders     = 5,
            title         = 'PyQt5 MultiSlider',
            window_width  = 500,
            window_height = None,
            x_offset      = 100,
            y_offset      = 100,
            label_width   = 200,
            slider_width  = 400,
            label_texts   = None,
            slider_mins   = None,
            slider_maxs   = None,
            slider_vals   = None,
            resolutions   = None,
            fontsize      = 10,
            verbose       = True,
        ):
        super().__init__()

        self.n_sliders     = n_sliders
        self.title         = title

        self.window_width  = window_width
        self.window_height = window_height if window_height else min(self.n_sliders * 60, 600)  # 최대 높이 제한
        self.x_offset      = x_offset
        self.y_offset      = y_offset
        
        self.label_width   = label_width
        self.slider_width  = slider_width

        if label_texts is None:
            self.label_texts = [f'Slider {i+1}' for i in range(n_sliders)]
        else:
            self.label_texts = label_texts
        self.slider_mins   = slider_mins
        self.slider_maxs   = slider_maxs
        self.slider_vals   = slider_vals
        if resolutions is None:
            self.resolutions   = [0.01]*self.n_sliders
        else:
            self.resolutions   = resolutions

        self.fontsize      = fontsize
        self.verbose       = verbose

        self.sliders        = []
        self.labels_widgets = []
        self.slider_values  = self.slider_vals.copy()

        self.init_ui()

    def init_ui(self):
        """
            Initialize UI
        """
        # Main layout
        main_layout = QVBoxLayout(self)

        # Make region scrollable 
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        # Widget
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)

        for i in range(self.n_sliders):
            # Slider
            slider = QSlider(Qt.Horizontal, self)

            # Resolution
            scale = 1 / self.resolutions[i]

            # Integer range
            int_min = int(self.slider_mins[i] * scale)
            int_max = int(self.slider_maxs[i] * scale)
            int_val = int(self.slider_vals[i] * scale)

            slider.setMinimum(int_min)
            slider.setMaximum(int_max)
            slider.setValue(int_val)
            slider.setSingleStep(1)
            slider.valueChanged.connect(lambda value, idx=i, s=scale: self.value_changed(idx, value, s))

            # Slider label
            label = QLabel(f'{self.label_texts[i]}: {self.slider_vals[i]:.4f}', self)
            label.setFixedWidth(self.label_width)
            label.setStyleSheet(f"font-size: {self.fontsize}px;")

            self.sliders.append(slider)
            self.labels_widgets.append(label)

            scroll_layout.addWidget(label, i, 0)
            scroll_layout.addWidget(slider, i, 1)

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)

        # Scrollable area
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)
        self.setWindowTitle(self.title)
        self.setGeometry(self.x_offset, self.y_offset, self.window_width, self.window_height)
        self.show()
        
        # Pause
        time.sleep(0.1)

    def value_changed(self, index, int_value, scale):
        # Change integer to float value
        float_value = int_value / scale
        self.slider_values[index] = float_value
        self.labels_widgets[index].setText(f'{self.label_texts[index]}: {float_value:.4f}')
        if self.verbose:
            print(f'Slider {index} Value: {float_value}')

    def get_slider_values(self):
        return self.slider_values

    def set_slider_values(self, slider_values):
        for i, val in enumerate(slider_values):
            scale = 1 / self.resolutions[i]
            int_val = int(val * scale)
            self.sliders[i].setValue(int_val)

    def set_slider_value(self, slider_idx, slider_value):
        scale = 1 / self.resolutions[slider_idx]
        int_val = int(slider_value * scale)
        self.sliders[slider_idx].setValue(int_val)

    def close(self):
        super().close()


def finite_difference_matrix(n, dt, order):
    """
    n: number of points
    dt: time interval
    order: (1=velocity, 2=acceleration, 3=jerk)
    """ 
    # Order
    if order == 1:  # velocity
        coeffs = np.array([-1, 1])
    elif order == 2:  # acceleration
        coeffs = np.array([1, -2, 1])
    elif order == 3:  # jerk
        coeffs = np.array([-1, 3, -3, 1])
    else:
        raise ValueError("Order must be 1, 2, or 3.")

    # Fill-in matrix
    mat = np.zeros((n, n))
    for i in range(n - order):
        for j, c in enumerate(coeffs):
            mat[i, i + j] = c

    # (optional) Handling boundary conditions with backward differences
    if order == 1:  # velocity
        mat[-1, -2:] = np.array([-1, 1])  # backward difference
    elif order == 2:  # acceleration
        mat[-1, -3:] = np.array([1, -2, 1])  # backward difference
        mat[-2, -3:] = np.array([1, -2, 1])  # backward difference
    elif order == 3:  # jerk
        mat[-1, -4:] = np.array([-1, 3, -3, 1])  # backward difference
        mat[-2, -4:] = np.array([-1, 3, -3, 1])  # backward difference
        mat[-3, -4:] = np.array([-1, 3, -3, 1])  # backward difference

    # Return 
    return mat / (dt ** order)

def get_A_vel_acc_jerk(n=100,dt=1e-2):
    """
        Get matrices to compute velocities, accelerations, and jerks
    """
    A_vel  = finite_difference_matrix(n,dt,order=1)
    A_acc  = finite_difference_matrix(n,dt,order=2)
    A_jerk = finite_difference_matrix(n,dt,order=3)
    return A_vel,A_acc,A_jerk

def smooth_optm_1d(
        traj,
        dt          = 0.1,
        x_init      = None,
        x_final     = None,
        vel_init    = None,
        vel_final   = None,
        x_lower     = None,
        x_upper     = None,
        vel_limit   = None,
        acc_limit   = None,
        jerk_limit  = None,
        idxs_remain = None,
        vals_remain = None,
        p_norm      = 2,
        verbose     = True,
    ):
    """
        1-D smoothing based on optimization
    """
    n = len(traj)
    A_pos = np.eye(n,n)
    A_vel,A_acc,A_jerk = get_A_vel_acc_jerk(n=n,dt=dt)
    
    # Objective 
    x = cp.Variable(n)
    objective = cp.Minimize(cp.norm(x-traj,p_norm))
    
    # Equality constraints
    A_list,b_list = [],[]
    if x_init is not None:
        A_list.append(A_pos[0,:])
        b_list.append(x_init)
    if x_final is not None:
        A_list.append(A_pos[-1,:])
        b_list.append(x_final)
    if vel_init is not None:
        A_list.append(A_vel[0,:])
        b_list.append(vel_init)
    if vel_final is not None:
        A_list.append(A_vel[-1,:])
        b_list.append(vel_final)
    if idxs_remain is not None:
        A_list.append(A_pos[idxs_remain,:])
        if vals_remain is not None:
            b_list.append(vals_remain)
        else:
            b_list.append(traj[idxs_remain])

    # Inequality constraints
    C_list,d_list = [],[]
    if x_lower is not None:
        C_list.append(-A_pos)
        d_list.append(-x_lower*np.ones(n))
    if x_upper is not None:
        C_list.append(A_pos)
        d_list.append(x_upper*np.ones(n))
    if vel_limit is not None:
        C_list.append(A_vel)
        C_list.append(-A_vel)
        d_list.append(vel_limit*np.ones(n))
        d_list.append(vel_limit*np.ones(n))
    if acc_limit is not None:
        C_list.append(A_acc)
        C_list.append(-A_acc)
        d_list.append(acc_limit*np.ones(n))
        d_list.append(acc_limit*np.ones(n))
    if jerk_limit is not None:
        C_list.append(A_jerk)
        C_list.append(-A_jerk)
        d_list.append(jerk_limit*np.ones(n))
        d_list.append(jerk_limit*np.ones(n))
    constraints = []
    if A_list:
        A = np.vstack(A_list)
        b = np.hstack(b_list).squeeze()
        constraints.append(A @ x == b) 
    if C_list:
        C = np.vstack(C_list)
        d = np.hstack(d_list).squeeze()
        constraints.append(C @ x <= d)
    
    # Solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    # Return
    traj_smt = x.value

    # Null check
    if traj_smt is None and verbose:
        print ("[smooth_optm_1d] Optimization failed.")
    return traj_smt

def smooth_gaussian_1d(traj,sigma=5.0,mode='nearest',radius=5):
    """ 
        Smooting using Gaussian filter
    """
    from scipy.ndimage import gaussian_filter1d
    traj_smt = gaussian_filter1d(
        input  = traj,
        sigma  = sigma,
        mode   = 'nearest',
        radius = int(radius),
    )
    return traj_smt
    
def plot_traj_vel_acc_jerk(
        t,
        traj,
        traj_smt = None,
        figsize  = (6,6),
        title    = 'Trajectory',
        ):
    """ 
        Plot trajectory, velocity, acceleration, and jerk
    """
    n  = len(t)
    dt = t[1]-t[0]
    # Compute velocity, acceleration, and jerk
    A_vel,A_acc,A_jerk = get_A_vel_acc_jerk(n=n,dt=dt)
    vel  = A_vel @ traj
    acc  = A_acc @ traj
    jerk = A_jerk @ traj
    if traj_smt is not None:
        vel_smt  = A_vel @ traj_smt
        acc_smt  = A_acc @ traj_smt
        jerk_smt = A_jerk @ traj_smt
    # Plot
    plt.figure(figsize=figsize)
    plt.subplot(4, 1, 1)
    plt.plot(t,traj,'.-',ms=1,color='k',lw=1/5,label='Trajectory')
    if traj_smt is not None:
        plt.plot(t,traj_smt,'.-',ms=1,color='r',lw=1/5,label='Smoothed Trajectory')
    plt.legend(fontsize=8,loc='upper right')
    plt.subplot(4, 1, 2)
    plt.plot(t,vel,'.-',ms=1,color='k',lw=1/5,label='Velocity')
    if traj_smt is not None:
        plt.plot(t,vel_smt,'.-',ms=1,color='r',lw=1/5,label='Smoothed Velocity')
    plt.legend(fontsize=8,loc='upper right')
    plt.subplot(4, 1, 3)
    plt.plot(t,acc,'.-',ms=1,color='k',lw=1/5,label='Acceleration')
    if traj_smt is not None:
        plt.plot(t,acc_smt,'.-',ms=1,color='r',lw=1/5,label='Smoothed Acceleration')
    plt.legend(fontsize=8,loc='upper right')
    plt.subplot(4, 1, 4)
    plt.plot(t,jerk,'.-',ms=1,color='k',lw=1/5,label='Jerk')
    if traj_smt is not None:
        plt.plot(t,jerk_smt,'.-',ms=1,color='r',lw=1/5,label='Smoothed Jerk')
    plt.legend(fontsize=8,loc='upper right')
    plt.suptitle(title,fontsize=10)
    plt.subplots_adjust(hspace=0.2,top=0.95)
    plt.show()

def kernel_se(X1,X2,hyp={'g':1,'l':1}):
    """
        Squared exponential (SE) kernel function
    """
    if len(X1.shape) == 1: X1 = X1.reshape(-1,1)
    if len(X2.shape) == 1: X2 = X2.reshape(-1,1)
    K = hyp['g']*np.exp(-cdist(X1,X2,'sqeuclidean')/(2*hyp['l']*hyp['l']))
    return K

def kernel_levse(X1,X2,L1,L2,hyp={'g':1,'l':1}):
    """
        Leveraged SE kernel function
    """
    K = hyp['g']*np.exp(-cdist(X1,X2,'sqeuclidean')/(2*hyp['l']*hyp['l']))
    L = np.cos(np.pi/2.0*cdist(L1,L2,'cityblock'))
    return np.multiply(K,L)

def safe_chol(A,max_iter=100,eps=1e-20,verbose=False):
    """ 
        Safe Cholesky decomposition
    """
    A_use = A.copy()
    for iter in range(max_iter):
        try:
            L = np.linalg.cholesky(A_use)
            if verbose:
                print ("[safe_chol] Cholesky succeeded. iter:[%d] eps:[%.2e]"%(iter,eps))
            return L 
        except np.linalg.LinAlgError:
            A_use = A_use + eps*np.eye(A.shape[0])
            eps *= 10
    print ("[safe_chol] Cholesky failed. iter:[%d] eps:[%.2e]"%(iter,eps))
    return None

def soft_squash(x,x_min=-1,x_max=+1,margin=0.1):
    """
        Soft squashing numpy array
    """
    def th(z,m=0.0):
        # thresholding function 
        return (m)*(np.exp(2/m*z)-1)/(np.exp(2/m*z)+1)
    x_in = np.copy(x)
    idxs_upper = np.where(x_in>(x_max-margin))
    x_in[idxs_upper] = th(x_in[idxs_upper]-(x_max-margin),m=margin) + (x_max-margin)
    idxs_lower = np.where(x_in<(x_min+margin))
    x_in[idxs_lower] = th(x_in[idxs_lower]-(x_min+margin),m=margin) + (x_min+margin)
    return x_in    

def soft_squash_multidim(
        x      = np.random.randn(100,5),
        x_min  = -np.ones(5),
        x_max  = np.ones(5),
        margin = 0.1,
    ):
    """
        Multi-dim version of 'soft_squash' function
    """
    x_squash = np.copy(x)
    dim      = x.shape[1]
    for d_idx in range(dim):
        x_squash[:,d_idx] = soft_squash(
            x=x[:,d_idx],x_min=x_min[d_idx],x_max=x_max[d_idx],margin=margin)
    return x_squash 

def get_idxs_closest_ndarray(ndarray_query,ndarray_domain):
    return [np.argmin(np.abs(ndarray_query-x)) for x in ndarray_domain]

def get_interp_const_vel_traj_nd(
        anchors, # [L x D]
        vel = 1.0,
        HZ  = 100,
        ord = np.inf,
    ):
    """
        Get linearly interpolated constant velocity trajectory
        Output is (times_interp,anchors_interp,times_anchor,idxs_anchor)
    """
    L = anchors.shape[0]
    D = anchors.shape[1]
    dists = np.zeros(L)
    for tick in range(L):
        if tick > 0:
            p_prev,p_curr = anchors[tick-1,:],anchors[tick,:]
            dists[tick] = np.linalg.norm(p_prev-p_curr,ord=ord)
    times_anchor = np.cumsum(dists/vel) # [L]
    L_interp     = int(times_anchor[-1]*HZ)
    times_interp = np.linspace(0,times_anchor[-1],L_interp) # [L_interp]
    anchors_interp  = np.zeros((L_interp,D)) # [L_interp x D]
    for d_idx in range(D): # for each dim
        anchors_interp[:,d_idx] = np.interp(times_interp,times_anchor,anchors[:,d_idx])
    idxs_anchor = get_idxs_closest_ndarray(times_interp,times_anchor)
    return times_interp,anchors_interp,times_anchor,idxs_anchor

def interpolate_and_smooth_nd(
        anchors, # List or [N x D]
        HZ             = 50,
        vel_init       = 0.0,
        vel_final      = 0.0,
        x_lowers       = None, # [D]
        x_uppers       = None, # [D]
        vel_limit      = None, # [1]
        acc_limit      = None, # [1]
        jerk_limit     = None, # [1]
        vel_interp_max = np.deg2rad(180),
        vel_interp_min = np.deg2rad(10),
        n_interp       = 10,
        verbose        = False,
    ):
    """ 
        Interpolate anchors and smooth [N x D] anchors
    """
    if isinstance(anchors, list):
        # If 'anchors' is given as a list, make it an ndarray
        anchors = np.vstack(anchors)
    
    D = anchors.shape[1]
    vels = np.linspace(start=vel_interp_max,stop=vel_interp_min,num=n_interp)
    for v_idx,vel_interp in enumerate(vels):
        # First, interploate
        times,traj_interp,times_anchor,idxs_anchor = get_interp_const_vel_traj_nd(
            anchors = anchors,
            vel     = vel_interp,
            HZ      = HZ,
        )
        dt = times[1] - times[0]
        # Second, smooth
        traj_smt = np.zeros_like(traj_interp)
        is_success = True
        for d_idx in range(D):
            traj_d = traj_interp[:,d_idx]
            if x_lowers is not None: x_lower_d = x_lowers[d_idx]
            else: x_lower_d = None
            if x_uppers is not None: x_upper_d = x_uppers[d_idx]
            else: x_upper_d = None
            traj_smt_d = smooth_optm_1d(
                traj        = traj_d,
                dt          = dt,
                idxs_remain = idxs_anchor,
                vals_remain = anchors[:,d_idx],
                vel_init    = vel_init,
                vel_final   = vel_final,
                x_lower     = x_lower_d,
                x_upper     = x_upper_d,
                vel_limit   = vel_limit,
                acc_limit   = acc_limit,
                jerk_limit  = jerk_limit,
                p_norm      = 2,
                verbose     = False,
            )
            if traj_smt_d is None:
                is_success = False
                break
            # Append
            traj_smt[:,d_idx] = traj_smt_d

        # Check success
        if is_success:
            if verbose:
                print ("Optimization succeeded. vel_interp:[%.3f]"%(vel_interp))
            return times,traj_interp,traj_smt,times_anchor
        else:
            if verbose:
                print (" v_idx:[%d/%d] vel_interp:[%.2f] failed."%(v_idx,n_interp,vel_interp))
    
    # Optimization failed
    if verbose:
        print ("Optimization failed.")
    return times,traj_interp,traj_smt,times_anchor

def check_vel_acc_jerk_nd(
        times, # [L]
        traj, # [L x D]
        verbose = True,
        factor  = 1.0,
    ):
    """ 
        Check velocity, acceleration, jerk of n-dimensional trajectory
    """
    L,D = traj.shape[0],traj.shape[1]
    A_vel,A_acc,A_jerk = get_A_vel_acc_jerk(n=len(times),dt=times[1]-times[0])
    vel_inits,vel_finals,max_vels,max_accs,max_jerks = [],[],[],[],[]
    for d_idx in range(D):
        traj_d = traj[:,d_idx]
        vel = A_vel @ traj_d
        acc = A_acc @ traj_d
        jerk = A_jerk @ traj_d
        vel_inits.append(vel[0])
        vel_finals.append(vel[-1])
        max_vels.append(np.abs(vel).max())
        max_accs.append(np.abs(acc).max())
        max_jerks.append(np.abs(jerk).max())

    # Print
    if verbose:
        print ("Checking velocity, acceleration, and jerk of a L:[%d]xD:[%d] trajectory (factor:[%.2f])."%
               (L,D,factor))
        for d_idx in range(D):
            print (" dim:[%d/%d]: v_init:[%.2e] v_final:[%.2e] v_max:[%.2f] a_max:[%.2f] j_max:[%.2f]"%
                   (d_idx,D,
                    factor*vel_inits[d_idx],factor*vel_finals[d_idx],
                    factor*max_vels[d_idx],factor*max_accs[d_idx],factor*max_jerks[d_idx])
                )
            
    # Return
    return vel_inits,vel_finals,max_vels,max_accs,max_jerks

def animate_chains_slider(
        env,
        secs,
        chains,
        transparent       = True,
        black_sky         = True,
        r_link            = 0.005,
        rgba_link         = (0.05,0.05,0.05,0.9),
        plot_joint        = True,
        plot_joint_axis   = True,
        plot_joint_sphere = False,
        plot_joint_name   = False,
        axis_len_joint    = 0.05,
        axis_width_joint  = 0.005,
        plot_rev_axis     = True,
    ):
    """ 
        Animate chains with slider
    """
    # Reset
    env.reset(step=True)
    
    # Initialize slider
    L = len(secs)
    sliders = MultiSliderClass(
        n_slider      = 2,
        title         = 'Slider Tick',
        window_width  = 900,
        window_height = 100,
        x_offset      = 100,
        y_offset      = 100,
        slider_width  = 600,
        label_texts   = ['tick','mode (0:play,1:slider,2:reverse)'],
        slider_mins   = [0,0],
        slider_maxs   = [L-1,2],
        slider_vals   = [0,1.0],
        resolutions   = [0.1,1.0],
        verbose       = False,
    )
    
    # Loop
    env.init_viewer(
        transparent = transparent,
        black_sky   = black_sky,
    )
    tick,mode = 0,'slider' # 'play' / 'slider'
    while env.is_viewer_alive():
        # Update
        env.increase_tick()
        sliders.update() # update slider
        chain = chains[tick]
        sec = secs[tick]

        # Mode change
        if sliders.get_slider_values()[1] == 0.0: mode = 'play'
        elif sliders.get_slider_values()[1] == 1.0: mode = 'slider'
        elif sliders.get_slider_values()[1] == 2.0: mode = 'reverse'

        # Render
        if env.loop_every(tick_every=20) or (mode=='play') or (mode=='reverse'):
            chain.plot_chain_mujoco(
                env,
                r_link            = r_link,
                rgba_link         = rgba_link,
                plot_joint        = plot_joint,
                plot_joint_axis   = plot_joint_axis,
                plot_joint_sphere = plot_joint_sphere,
                plot_joint_name   = plot_joint_name,
                axis_len_joint    = axis_len_joint,
                axis_width_joint  = axis_width_joint,
                plot_rev_axis     = plot_rev_axis,
            )
            env.plot_T(p=np.array([0,0,0]),R=np.eye(3,3))
            # env.plot_time(p=np.array([0,0,1]),post_str=' mode:[%s]'%(mode))
            env.plot_text(
                p     = np.array([0,0,1]),
                label = '[%d] tick:[%d] time:[%.2f]sec mode:[%s]'%(env.tick,tick,sec,mode)
            )
            env.render()        

        # Proceed
        if mode == 'play':
            if tick < len(chains)-1: tick = tick + 1
            sliders.set_slider_value(slider_idx=0,slider_value=tick)
        elif mode == 'slider':
            tick = int(sliders.get_slider_values()[0])
        elif mode == 'reverse':
            if tick > 0: tick = tick - 1
            sliders.set_slider_value(slider_idx=0,slider_value=tick)
            
    # Close viewer and slider
    env.close_viewer() 
    sliders.close()
    
def animate_env_qpos_list(
        env,
        secs,
        qpos_list,
        viewer_title      = '',
        plot_contact_info = True,
        transparent       = True,
        black_sky         = True,
    ):
    """ 
        Animate env with 
    """
    # Reset
    env.reset(step=True)
    # Initialize slider
    L = len(secs)
    sliders = MultiSliderClass(
        n_slider      = 2,
        title         = 'Slider Tick',
        window_width  = 900,
        window_height = 100,
        x_offset      = 100,
        y_offset      = 100,
        slider_width  = 600,
        label_texts   = ['tick','mode (0:play,1:slider,2:reverse)'],
        slider_mins   = [0,0],
        slider_maxs   = [L-1,2],
        slider_vals   = [0,1.0],
        resolutions   = [0.1,1.0],
        verbose       = False,
    )
    # Loop
    env.init_viewer(
        transparent = transparent,
        title       = viewer_title,
        black_sky   = black_sky,
    )
    tick,mode = 0,'slider' # 'play' / 'slider'
    while env.is_viewer_alive():
        # Update
        # env.increase_tick()
        sliders.update() # update slider
        qpos = qpos_list[tick]
        env.forward(q=qpos)
        sec = secs[tick]

        # Mode change
        if sliders.get_slider_values()[1] == 0.0: mode = 'play'
        elif sliders.get_slider_values()[1] == 1.0: mode = 'slider'
        elif sliders.get_slider_values()[1] == 2.0: mode = 'reverse'

        # Render
        if env.loop_every(tick_every=20) or (mode=='play') or (mode=='reverse'):
            env.plot_T(p=np.array([0,0,0]),R=np.eye(3,3))
            env.viewer.add_overlay(loc='bottom left',text1='tick',text2='%d'%(env.tick))
            env.viewer.add_overlay(loc='bottom left',text1='sim time (sec)',text2='%.2f'%(sec))
            env.viewer.add_overlay(loc='bottom left',text1='mode',text2='%s'%(mode))
            if plot_contact_info:
                env.plot_contact_info()
            env.render()

        # Proceed
        if mode == 'play':
            if tick < len(qpos_list)-1: tick = tick + 1
            sliders.set_slider_value(slider_idx=0,slider_value=tick)
            if tick == (L-1): mode = 'slider'
        elif mode == 'slider':
            tick = int(sliders.get_slider_values()[0])
        elif mode == 'reverse':
            if tick > 0: tick = tick - 1
            sliders.set_slider_value(slider_idx=0,slider_value=tick)
            
    # Close viewer and slider
    env.close_viewer() 
    sliders.close()
        
def np_uv(vec):
    """
        Get unit vector
    """
    x = np.array(vec)
    len = np.linalg.norm(x)
    if len <= 1e-6:
        return np.array([0,0,1])
    else:
        return x/len    
    
def uv_T_joi(T_joi,joi_fr,joi_to):
    """ 
        Get unit vector between to JOI poses
    """
    return np_uv(t2p(T_joi[joi_to]) - t2p(T_joi[joi_fr]))

def len_T_joi(T_joi,joi_fr,joi_to):
    """ 
        Get length between two JOI poses
    """
    return np.linalg.norm(t2p(T_joi[joi_to]) - t2p(T_joi[joi_fr]))

def get_consecutive_subarrays(array,min_element=1):
    """ 
        Get consecutive sub arrays from an array
    """
    split_points = np.where(np.diff(array) != 1)[0] + 1
    subarrays = np.split(array,split_points)    
    return [subarray for subarray in subarrays if len(subarray) >= min_element]

def load_image(image_path):
    """ 
        Load image to ndarray (unit8)
    """
    return np.array(Image.open(image_path))

def imshows(img_list,title_list,figsize=(8,2),fontsize=8):
    """ 
        Plot multiple images in a row
    """
    n_img = len(img_list)
    plt.figure(figsize=(8,2))
    for img_idx in range(n_img):
        img   = img_list[img_idx]
        title = title_list[img_idx]
        plt.subplot(1,n_img,img_idx+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(title,fontsize=fontsize)
    plt.show()
    
def depth_to_gray_img(depth,max_val=10.0):
    """
        1-channel float-type depth image to 3-channel unit8-type gray image
    """
    depth_clip = np.clip(depth,a_min=0.0,a_max=max_val) # float-type
    img = np.tile(255*depth_clip[:,:,np.newaxis]/depth_clip.max(),(1,1,3)).astype(np.uint8) # unit8-type
    return img

def get_monitor_size():
    """ 
        Get monitor size
    """
    w,h = pyautogui.size()
    return w,h
    
def get_xml_string_from_path(xml_path):
    # Parse the XML file
    tree = ET.parse(xml_path)
    
    # Get the root element of the XML
    root = tree.getroot()
    
    # Convert the ElementTree object to a string
    xml_string = ET.tostring(root, encoding='unicode', method='xml')
    
    return xml_string

def printmd(string):
    display(Markdown(string)) 
    
def prettify(elem):
    """
        Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ")
    
    # 불필요한 공백 제거 (빈 줄)
    lines = [line for line in pretty_xml.splitlines() if line.strip()]
    return "\n".join(lines)
