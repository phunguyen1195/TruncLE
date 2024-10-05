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

class Lorenzle(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=10.0, alpha=0.01, reward_type='precal', max = False, sphere_R = 0.2):
        super(Lorenzle, self).__init__() 

        self.collected_states = list()
        self.goal_state = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if isinstance(alpha, float):
            self.alpha = alpha
        elif isinstance(alpha, dict):
            self.alpha1 = alpha['alpha1']
            self.alpha2 = alpha['alpha2']


        self.precal_le_ = np.load("Precal_le/precal_lorenz_full_0177_0.001_350.npy")
        self.precal_points = np.load("Precal_le/precal_lorenz_points_full_0177_0.001_350.npy")

        self.get_spatial = spatial.KDTree(self.precal_points)

        self.max = max
        self.reward_type = reward_type
        self.infinite = True
        self.T = 500
        self.close_to_goal = True
        self.sphere_R = sphere_R

        self.sigma = 16.0
        self.r = 45.92
        self.b = 4.0

        self.C = ((self.b ** 2) * ((self.sigma + self.r) ** 2)) / (4 * (self.b - 1))
        self.alpha = alpha
        self.dt = 0.01
        self.dyn = {"sigma":self.sigma , "R":self.r, "b": self.b}
        self.render_mode = render_mode
        #self.renderer = Renderer(self.render_mode, self._render)
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        self.action_impact = 10
        self.action_space = spaces.Box(low= - self.action_impact, high= self.action_impact, shape = (1,),dtype=np.float32)


        self.high_env_range = np.array([25.0, 25.0, 77.0], dtype=np.float32)
        self.low_env_range = np.array([-25.0, -25.0, -0.1], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=self.low_env_range, high=self.high_env_range, shape = (3,),dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def linearized_lorenz (self, x0, dyn, y_lorenz):
        sigma = dyn['sigma']
        R = dyn['R']
        b = dyn['b']
        x = x0[0]
        y = x0[1]
        z = x0[2]
        return np.array([sigma * (y - x), 
                        (R - y_lorenz[2])*x - y - y_lorenz[0]*z,
                        y_lorenz[1]*x + y_lorenz[0]*y - b*z])



    def lorenz (self, x0, dyn, action):
            
        sigma = dyn['sigma']
        R = dyn['R']
        b = dyn['b']
        x = x0[0]
        y = x0[1]
        z = x0[2]
        return np.array([sigma * (y - x) + action[0], 
                        x * (R - z) - y + action[0], 
                        x * y - b * z + action[0]])

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
        # sum positive
            le = sum(le[le>0])
        # max
        else:
            le = max(le)

        return le

    def step(self, u):
        x = self.state  # th := theta

        dyn = self.dyn
        f = self.lorenz
        dt = self.dt

        newx, v = self.f_x ( dyn, f, dt, x, u)

        newx[0] = bound(newx[0], -25.0, 25.0)
        newx[1] = bound(newx[1], -25.0, 25.0)
        newx[2] = bound(newx[2], -0.1, 77)

        if self.reward_type == 'precal':
            self.cost = self.precal_le(newx)

        elif self.reward_type == 'quadratic':
            self.cost = -( newx[0] ** 2 + self.alpha1 * newx[1] ** 2 + self.alpha2 * newx[2] ** 2)
        elif self.reward_type == 'sparse':
            if np.linalg.norm(newx - self.goal_state) ** 2 - 0.001 ** 2 > 0.0:
                self.cost = 0
            else:
                self.cost = 1

        terminated = False


        self.state = newx

        return self._get_obs(), self.cost, terminated, {"mean x": x[0], "mean y": x[1], "mean z": x[2], 'trajectory': x, 'action': u}

    def reset(self, obs = None):
        #put even closer to reward
        if self.close_to_goal == True:
            high = np.array([self.goal_state[0] + self.sphere_R,self.goal_state[1]+ self.sphere_R, 25], dtype=np.float32)
            low = np.array([self.goal_state[0]- self.sphere_R,self.goal_state[1]- self.sphere_R, -0.1], dtype=np.float32)  # We enforce symmetric limits.

            self.state = self.np_random.uniform(low=low, high=high)

        elif obs != None:
            self.state = obs
            # print (self.state)
        else:
            high = self.high_env_range
            low = self.low_env_range  # We enforce symmetric limits.

            self.state = self.np_random.uniform(low=low, high=high)

        self.last_u = None
        return self._get_obs()


    def _get_obs(self):
        obsv = self.state
        return np.array(obsv, dtype=np.float32)

    def render(self, mode="human"):
        pass

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