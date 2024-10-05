from os import path
from typing import Optional

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
#from gym.utils.renderer import Renderer
from gym.error import DependencyNotInstalled
#from gym.utils.renderer import Renderer
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy import spatial


plt.style.use('fivethirtyeight')

class pendulum_le(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=9.81, alpha = 0.1, reward_type='precal', max = False):
        super(pendulum_le, self).__init__()


        self.close_to_goal = False
        self.precal_le_  = np.load("Precal_le/precal_pendulum_0.001_400.npy")
        self.precal_points = np.load("Precal_le/precal_pendulum_points_0.001_400.npy")
        self.get_spatial = spatial.KDTree(self.precal_points)
        self.goal_state = np.array([np.pi, 0.0])
        self.max_speed = 8
        self.max_torque = 5
        self.T = 200
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.dt = 0.1
        self.dyn = {"g":self.g , "m":self.m, "l": self.l}
        self.alpha = alpha
        self.max = max
        self.reward_type = reward_type
        if isinstance(alpha, float):
            self.alpha = alpha
        elif isinstance(alpha, dict):
            self.alpha1 = alpha['alpha1']

        self.render_mode = render_mode
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        self.high_env_range = np.array([np.pi+np.pi/64, self.max_speed], dtype=np.float32)
        self.low_env_range = np.array([np.pi-np.pi/64, -self.max_speed], dtype=np.float32)
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        low = np.array([-1.0, -1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape = (1,),dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape = (3,), dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def simple_pendulum (self, x0, dyn, action):
        g = dyn['g'] 
        l = dyn['l']
        m = dyn['m']
        return np.array([x0[1] , (-g/l)*np.sin(x0[0]) + (1.0 / m) * action[0]])

    def linearized_simple_pendulum (self, x0, dyn, y_pendulum):
        g = dyn['g'] 
        l = dyn['l']
        return np.array([0*x0[0] + 1*x0[1],
                        (-g/l)*np.cos(y_pendulum[0])*x0[0] + 0*x0[1]])

    def RungeKutta (self, dyn, f, dt, x0, action):

        k1 = f(x0.copy(), dyn, action/4.0) #[x,y,z]*0.1 example
        k2 = f(x0.copy() + 0.5*k1*dt, dyn, action/4.0)
        k3 = f(x0.copy() + 0.5*k2*dt, dyn, action/4.0)
        k4 = f(x0.copy() + k3*dt, dyn, action/4.0)

        x = x0 + ((k1 + 2*k2 + 2*k3 + k4)/6) * dt

        return x, k4
    
    def RungeKutta_linearized (self, dyn, f, dt, x0, y):
        k1 = f(x0, dyn, y) #[x,y,z]*0.1 example
        k2 = f(x0+0.5*k1*dt,dyn, y)
        k3 = f(x0 + 0.5*k2*dt, dyn, y)
        k4 = f(x0 + k3*dt, dyn, y)
        
        x = x0 + ((k1 + 2*k2 + 2*k3 + k4)/6) * dt

        return x


    def f_x (self, dyn, f, dt, x0, action):
        #change to get one x sample at a time
        x, v = self.RungeKutta(dyn, f, dt, x0, action)
        return x, v

    
    def precal_le(self, s):
        
        index = self.get_spatial.query(s)[1]
        le = self.precal_le_[index]
        if self.max == False:
            le = sum(le[le>0])
        else:
            le = max(le)
        return le

    
    def step(self, u):
        x = self.state  # th := theta

        dyn = self.dyn
        f = self.simple_pendulum
        dt = self.dt
        newx, v = self.f_x ( dyn, f, dt, x, u)
        newx[0] = wrap(newx[0], 0, 2*np.pi)
        newx[1] = bound(newx[1], -8, 8)
        if self.reward_type == 'precal':
            self.cost = self.precal_le(newx) + self.alpha * u[0] ** 2
        elif self.reward_type == 'quadratic':
            self.cost = - ((np.pi-newx[0])**2 + self.alpha1*newx[1]**2)
        elif self.reward_type == 'sparse':
            if  - np.cos(newx[0]) >= 1-0.001:
                self.cost = 1
            else:
                self.cost = 0
        terminated = False
        self.state = newx

        return self._get_obs(), self.cost, terminated, {'trajectory': x, 'action': u, "angle": x[0], "velocity": x[1]}

    def reset(self):

        if self.close_to_goal == True:
            self.state = self.np_random.uniform(low=self.goal_state-0.5, high=self.goal_state+0.5)
        else:
            temp_state = self.np_random.uniform(low=self.low_env_range, high=self.high_env_range)
            if temp_state[0] < np.pi:
                temp_state[0] = np.pi + temp_state[0]
            else:
                temp_state[0] = temp_state[0] - np.pi
            self.state = temp_state
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
        # return np.array(obsv, dtype=np.float32)

    def render(self, mode="human"):
        pass


def wrap(x, m, M):
    """Wraps ``x`` so m <= x <= M; but unlike ``bound()`` which
    truncates, ``wrap()`` wraps x around the coordinate system defined by m,M.\n
    For example, m = -180, M = 180 (degrees), x = 360 --> returns 0.

    Args:
        x: a scalar
        m: minimum possible value in range
        M: maximum possible value in range

    Returns:
        x: a scalar, wrapped
    """
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


def bound(x, m, M=None):
    """Either have m as scalar, so bound(x,m,M) which returns m <= x <= M *OR*
    have m as length 2 vector, bound(x,m, <IGNORED>) returns m[0] <= x <= m[1].

    Args:
        x: scalar
        m: The lower bound
        M: The upper bound

    Returns:
        x: scalar, bound between min (m) and Max (M)
    """
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)
#empowerment guy