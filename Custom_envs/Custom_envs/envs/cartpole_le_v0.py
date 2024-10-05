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

class cartpole_le(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=9.81, alpha=0.01, reward_type='precal', max = False):
        super(cartpole_le, self).__init__()


        self.close_to_goal = False

        self.T = 350
        self.max_torque = 8.0
        self.dt = 0.1
        self.dyn = {"g":9.8, "m2": 1.0, "l": 1.0, "m1": 10.0}
        self.alpha = alpha
        self.reward_type = reward_type
        if isinstance(alpha, float):
            self.alpha = alpha
        elif isinstance(alpha, dict):
            self.alpha1 = alpha['alpha1']

        
        self.max = max
        self.render_mode = render_mode
        # /home/015970994/masterchaos/precal_le/New_Cartpole/cartpole_052724/precal_cartpole_0.001_400.npy
        # self.precal_le_ = np.loadtxt("precal_le_02.txt", delimiter=',')
        # self.precal_points = np.load("points_02.npy")
        # self.precal_le_ = np.load("/home/015970994/masterchaos/precal_le/New_Cartpole/full/precal_cartpole_full_40_0.001_400.npy")
        # self.precal_points = np.load("/home/015970994/masterchaos/precal_le/New_Cartpole/full/precal_cartpole_points_full_40_0.001_400.npy")

        self.precal_le_ = np.load("Precal_le/precal_cartpole_0.001_400.npy")
        self.precal_points = np.load("Precal_le/precal_cartpole_points_0.001_400.npy")
        self.precal_points = self.precal_points[:,2:4]
        self.get_spatial = spatial.KDTree(self.precal_points)
        self.trajectory_collection = []
        #self.renderer = Renderer(self.render_mode, self._render)
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        # self.high_env_range = np.array([2,1, 0+ 5*np.pi/16, 8], dtype=np.float32)
        # self.low_env_range = np.array([-2,-1, 0- 5*np.pi/16, -8], dtype=np.float32)
        self.high_env_range = np.array([2,1, 0+ np.pi/2, 8], dtype=np.float32)
        self.low_env_range = np.array([-2,-1, 0- np.pi/2, -8], dtype=np.float32)
        high = np.array([2.0, 1.0, 1.0, 1.0, 8.0], dtype=np.float32)
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape = (1,),dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, shape = (5,), dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def cartpole (self, x0, dyn, action):
        g = dyn['g'] 
        l = dyn['l']
        m1 = dyn['m1']
        m2 = dyn['m2']
        #print (x0)
        f = np.array([x0[1],
                        (0.011*(x0[3]**2)*np.sin(x0[2]) + 0.098*np.cos(x0[2])*np.sin(x0[2])) / (0.01*(np.cos(x0[2])**2) - 0.22),
                        x0[3],
                        (-1.96*np.sin(x0[2]) - 0.01*(x0[3]**2)*np.cos(x0[2])*np.sin(x0[2])) / (0.01*(np.cos(x0[2])**2) - 0.22)
                        ], dtype=np.float32)
        g = np.array([0,
                    0.11 / (0.01*(np.cos(x0[2])**2) - 0.22),
                    0,
                    (-0.1*np.cos(x0[2])) / (0.01*(np.cos(x0[2])**2) - 0.22)
            
        ], dtype=np.float32)
        return f + g*action[0]
    
    def linearized_cartpole (self, x0, dyn, y_cartpole):
        g = dyn['g'] 
        l = dyn['l']
        m1 = dyn['m1']
        m2 = dyn['m2']
        led_cartpole = np.array([[0, 1, 0, 0],
                                [0, 0, (0.098*(np.cos(y_cartpole[2])**2) 
                                        + 0.011*(y_cartpole[3]**2)*np.cos(y_cartpole[2]) 
                                        - 0.098*(np.sin(y_cartpole[2])**2)) / (0.01*(np.cos(y_cartpole[2])**2) - 0.22) 
                                + 0.02*((0.011*(y_cartpole[3]**2)*np.sin(y_cartpole[2]) 
                                        + 0.098*np.cos(y_cartpole[2])*np.sin(y_cartpole[2])) / 
                                        ((0.01*(np.cos(y_cartpole[2])**2) - 0.22)**2))*np.cos(y_cartpole[2])*np.sin(y_cartpole[2]), 
                                (0.022*np.sin(y_cartpole[2])*y_cartpole[3]) / (0.01*(np.cos(y_cartpole[2])**2) - 0.22)],
                                [0, 0, 0, 1],
                                [0, 0, (0.01*(np.sin(y_cartpole[2])**2)*(y_cartpole[3]**2) - 1.96*np.cos(y_cartpole[2]) - 0.01*(np.cos(y_cartpole[2])**2)*(y_cartpole[3]**2)) / (0.01*(np.cos(y_cartpole[2])**2) - 0.22) + 0.02*((-1.96*np.sin(y_cartpole[2]) - 0.01*(y_cartpole[3]**2)*np.cos(y_cartpole[2])*np.sin(y_cartpole[2])) / ((0.01*(np.cos(y_cartpole[2])**2) - 0.22)**2))*np.cos(y_cartpole[2])*np.sin(y_cartpole[2]), (-0.02*np.cos(y_cartpole[2])*np.sin(y_cartpole[2])*y_cartpole[3]) / (0.01*(np.cos(y_cartpole[2])**2) - 0.22)]
                                ], dtype=np.float32)
        v = np.dot(led_cartpole,x0)
        return v

    def RungeKutta (self, dyn, f, dt, x0, action):

        k1 = f(x0.copy(), dyn, action/4.0) #[x,y,z]*0.1 example
        k2 = f(x0.copy() + 0.5*k1*dt, dyn, action/4.0)
        k3 = f(x0.copy() + 0.5*k2*dt, dyn, action/4.0)
        k4 = f(x0.copy() + k3*dt, dyn, action/4.0)

        x = x0 + ((k1 + 2*k2 + 2*k3 + k4)/6) * dt

        return x
    
    def RungeKutta_linearized (self, dyn, f, dt, x0, y):
        k1 = f(x0, dyn, y) #[x,y,z]*0.1 example
        k2 = f(x0+0.5*k1*dt,dyn, y)
        k3 = f(x0 + 0.5*k2*dt, dyn, y)
        k4 = f(x0 + k3*dt, dyn, y)
        
        x = x0 + ((k1 + 2*k2 + 2*k3 + k4)/6) * dt

        return x


    def f_x (self, dyn, f, dt, x0, action):
        #change to get one x sample at a time
        x = self.RungeKutta(dyn, f, dt, x0, action)
        #x_noaction = self.RungeKutta(dyn, f, dt, x0, np.array([0.0, 0.0, 0.0]))
        return x
    
    
    def precal_le(self, s):
            
            # s[0] = np.cos(s[0])
            # s[2] = np.cos(s[2])
        s_temp = s[2:4]
        # s_temp = s
        index = self.get_spatial.query(s_temp)[1]
        le = self.precal_le_[index]

        # print ('state:', s)
        # print ('nearest point:', self.precal_points[index])
        # print ('le at point:', le)
        if self.max == False:
        # sum positive
            le = sum(le[le>0])
        # max
        else:
            le = max(le)

        # print ("le:", le, "state temp", s_temp, "state", s)
        return le
    

    def step(self, u):
        x = self.state  # th := theta

        dyn = self.dyn
        f = self.cartpole
        dt = self.dt
        # u = np.clip(u, -self.max_torque, self.max_torque)
        newx = self.f_x ( dyn, f, dt, x, u)
        newx[2] = wrap(newx[2], -np.pi, np.pi)
        newx[3] = bound(newx[3], -8, 8)
        newx[0] = bound(newx[0], -2, 2)

        
        if self.reward_type == 'precal':
            self.cost = self.precal_le(newx)  + self.alpha * u[0] ** 2
        elif self.reward_type == 'quadratic':
            self.cost = - (newx[2]**2 + self.alpha1*newx[3]**2)
        elif self.reward_type == 'sparse':
            if  np.cos(newx[2]) >= 1-0.001:
                self.cost = 1
            else:
                self.cost = 0

        terminated = False

        self.state = newx

        return self._get_obs(), self.cost, terminated, {'angle': x[2], 'velocity':x[3], 'action': u[0], 'trajectory': np.array([x[0],x[1],x[2],x[3]])}

    def reset(self):
        #put even closer to reward
        # high = np.array([np.pi, 1.0])
        # self.state = self.np_random.uniform(low=-high, high=high)
        if self.close_to_goal == True:
            sample_state_low = np.array([-2 , -1, 0.0-3*np.pi/4, -8])
            sample_state_hign = np.array([2 , 1, 0.0+3*np.pi/4, 8])
            self.state = self.np_random.uniform(low=sample_state_low, high=sample_state_hign)
        else:
            temp_state = self.np_random.uniform(low=self.low_env_range, high=self.high_env_range)
            if temp_state[2] < 0:
                temp_state[2] = np.pi + temp_state[2]
            else:
                temp_state[2] = temp_state[2] - np.pi
            self.state = temp_state
        #self.collected_states = self.collected_states.append(self.state)
        self.last_u = None

        return self._get_obs()


    def _get_obs(self):
        position, velocity, theta, thetadot = self.state
        #self.collected_states.append(self.state)
        return np.array([position, velocity, np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
        # return np.array(obsv, dtype=np.float32)

    def render(self, mode="human"):
        pass

#empowerment guy
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