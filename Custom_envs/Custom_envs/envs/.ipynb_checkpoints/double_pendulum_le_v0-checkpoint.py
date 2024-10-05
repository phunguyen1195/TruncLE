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

class double_pendulum_le(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=9.81):
        super(double_pendulum_le, self).__init__()

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        self.ax.scatter(0,0, s=100, color="red")
        self.close_to_goal = False
        self.infinite = True
        self.precal_le_  = np.load("/home/015970994/masterchaos/precal_le/Double_pendulum/precal_dp_26.npy")
        self.precal_points = np.load("/home/015970994/masterchaos/precal_le/Double_pendulum/precal_dp_points_26.npy")
        self.get_spatial = spatial.KDTree(self.precal_points)
        self.goal_state = np.array([np.pi, 0.0])
        # self.max_speed = 8
        # self.max_torque = 0.1
        self.max_torque = 1
        self.T = 500
        self.g = g
        self.m1 = 1.0
        self.l = 1.0
        self.m2 = 1.0
        self.dt = 0.01
        self.dyn = {"g":self.g , "m1":self.m1, "l": self.l, "m2": self.m2}
        self.alpha = 10
        self.render_mode = render_mode
        #self.renderer = Renderer(self.render_mode, self._render)
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True
        #increase velocities
        # self.high_env_range = np.array([np.pi/2+0.5, 0+8, 0+0.5,0+8], dtype=np.float32)
        # self.low_env_range = np.array([np.pi/2-0.5, 0-8, 0-0.5,0-8], dtype=np.float32)

        self.low_env_range = np.array([np.pi-np.pi/4, -4,-np.pi/4,-4])
        self.high_env_range = np.array([np.pi+np.pi/4,4,np.pi/4,4])

        # high = np.array(
        #     [1.0, 1.0, 1.0, 1.0, 4 * pi, 9 * pi], dtype=np.float32
        # )
        # low = -high

        high = np.array(
            [np.cos(np.pi+np.pi/4), np.sin(np.pi+np.pi/4),4, np.cos(np.pi/4), np.sin(np.pi/4), 4], dtype=np.float32
        )
        low = np.array(
            [np.cos(np.pi-np.pi/4), np.sin(np.pi-np.pi/4),-4, np.cos(-np.pi/4), np.sin(-np.pi/4), -4], dtype=np.float32
        )
        
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape = (1,),dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, shape = (6,), dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    # def double_pendulum (self, x0, dyn, action):
    #     g = dyn['g']
    #     l = dyn['l']
    #     m1 = dyn['m1']
    #     m2 = dyn['m2']
    #     #print (x0)
    #     f = np.array([x0[1],
    #                 ((-(1.25 + 0.5*np.cos(x0[2]))*(((1.25 + 0.5*np.cos(x0[2]))*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])) / (3.5 + np.cos(x0[2])) - 4.9*np.cos(x0[0] + x0[2]) - 0.5*(x0[1]**2)*np.sin(x0[2]))) / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2]))) + 0.5*(x0[3]**2)*np.sin(x0[2]) + np.sin(x0[2])*x0[1]*x0[3] - 4.9*np.cos(x0[0] + x0[2]) - 14.7*np.cos(x0[0])) / (3.5 + np.cos(x0[2])),
    #                 x0[3],
    #                 (((1.25 + 0.5*np.cos(x0[2]))*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])) / (3.5 + np.cos(x0[2])) - 4.9*np.cos(x0[0] + x0[2]) - 0.5*(x0[1]**2)*np.sin(x0[2])) / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2])))
    #                 ], dtype=np.float32)
    #     g = np.array([0,
    #         (-1.25 - 0.5*np.cos(x0[2])) / ((1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2])))*(3.5 + np.cos(x0[2]))),
    #         0,
    #         1 / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2])))
    #         ], dtype=np.float32)
    #     return f + g*action[0]

    def double_pendulum (self, x0, dyn, dt, action):
        g = dyn['g']
        l = dyn['l']
        m1 = dyn['m1']
        m2 = dyn['m2']
        #print (x0)
        f = np.array([x0[1],
                    ((-(1.25 + 0.5*np.cos(x0[2]))*(((1.25 + 0.5*np.cos(x0[2]))*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])) / (3.5 + np.cos(x0[2])) - 4.9*np.cos(x0[0] + x0[2]) - 0.5*(x0[1]**2)*np.sin(x0[2]))) / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2]))) + 0.5*(x0[3]**2)*np.sin(x0[2]) + np.sin(x0[2])*x0[1]*x0[3] - 4.9*np.cos(x0[0] + x0[2]) - 14.7*np.cos(x0[0])) / (3.5 + np.cos(x0[2])),
                    x0[3],
                    (((1.25 + 0.5*np.cos(x0[2]))*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])) / (3.5 + np.cos(x0[2])) - 4.9*np.cos(x0[0] + x0[2]) - 0.5*(x0[1]**2)*np.sin(x0[2])) / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2])))
                    ], dtype=np.float32)
        g = np.array([0,
            (-1.25 - 0.5*np.cos(x0[2])) / ((1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2])))*(3.5 + np.cos(x0[2]))),
            0,
            1 / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2])))
            ], dtype=np.float32)
        return x0 + f*dt + g*dt*action[0]

    def linearized_double_pendulum (self, identity, dyn, x0):
        g = dyn['g']
        l = dyn['l']
        m1 = dyn['m1']
        m2 = dyn['m2']
        led_double_pendulum = np.array([[0, 1, 0, 0],
                                [(((-1.25 - 0.5*np.cos(x0[2]))*(((1.25 + 0.5*np.cos(x0[2]))*(-4.9*np.sin(x0[0] + x0[2]) - 14.7*np.sin(x0[0]))) / (3.5 + np.cos(x0[2])) + 4.9*np.sin(x0[0] + x0[2]))) / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2]))) + 4.9*np.sin(x0[0] + x0[2]) + 14.7*np.sin(x0[0])) / (3.5 + np.cos(x0[2])),
                                (((-1.25 - 0.5*np.cos(x0[2]))*((-(1.25 + 0.5*np.cos(x0[2]))*np.sin(x0[2])*x0[3]) / (3.5 + np.cos(x0[2])) - np.sin(x0[2])*x0[1])) / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2]))) + np.sin(x0[2])*x0[3]) / (3.5 + np.cos(x0[2])),
                                (((-1.25 - 0.5*np.cos(x0[2]))*(((1.25 + 0.5*np.cos(x0[2]))*(-4.9*np.sin(x0[0] + x0[2]) - 0.5*(x0[3]**2)*np.cos(x0[2]) - np.cos(x0[2])*x0[1]*x0[3]) - 0.5*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])*np.sin(x0[2])) / (3.5 + np.cos(x0[2])) + 4.9*np.sin(x0[0] + x0[2]) + (((1.25 + 0.5*np.cos(x0[2]))*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])) / ((3.5 + np.cos(x0[2]))**2))*np.sin(x0[2]) - 0.5*(x0[1]**2)*np.cos(x0[2])) - 0.5*((-(1.25 + 0.5*np.cos(x0[2]))*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])) / (3.5 + np.cos(x0[2])) + 4.9*np.cos(x0[0] + x0[2]) + 0.5*(x0[1]**2)*np.sin(x0[2]))*np.sin(x0[2])) / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2]))) + 4.9*np.sin(x0[0] + x0[2]) + 0.5*(x0[3]**2)*np.cos(x0[2]) + np.cos(x0[2])*x0[1]*x0[3] - (((-((1.25 + 0.5*np.cos(x0[2]))**2)) / ((3.5 + np.cos(x0[2]))**2))*np.sin(x0[2]) + (0.5*(2.5 + np.cos(x0[2]))*np.sin(x0[2])) / (3.5 + np.cos(x0[2])))*((-(1.25 + 0.5*np.cos(x0[2]))*(((1.25 + 0.5*np.cos(x0[2]))*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])) / (3.5 + np.cos(x0[2])) - 4.9*np.cos(x0[0] + x0[2]) - 0.5*(x0[1]**2)*np.sin(x0[2]))) / ((1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2])))**2))) / (3.5 + np.cos(x0[2])) + (((-(1.25 + 0.5*np.cos(x0[2]))*(((1.25 + 0.5*np.cos(x0[2]))*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])) / (3.5 + np.cos(x0[2])) - 4.9*np.cos(x0[0] + x0[2]) - 0.5*(x0[1]**2)*np.sin(x0[2]))) / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2]))) + 0.5*(x0[3]**2)*np.sin(x0[2]) + np.sin(x0[2])*x0[1]*x0[3] - 4.9*np.cos(x0[0] + x0[2]) - 14.7*np.cos(x0[0])) / ((3.5 + np.cos(x0[2]))**2))*np.sin(x0[2]),
                                (((-1.25 - 0.5*np.cos(x0[2]))*(1.25 + 0.5*np.cos(x0[2]))*(-np.sin(x0[2])*x0[1] - np.sin(x0[2])*x0[3])) / ((1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2])))*(3.5 + np.cos(x0[2]))) + np.sin(x0[2])*x0[1] + np.sin(x0[2])*x0[3]) / (3.5 + np.cos(x0[2]))],
                                        [0, 0, 0, 1],
                                [(((1.25 + 0.5*np.cos(x0[2]))*(-4.9*np.sin(x0[0] + x0[2]) - 14.7*np.sin(x0[0]))) / (3.5 + np.cos(x0[2])) + 4.9*np.sin(x0[0] + x0[2])) / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2]))),
                                ((-(1.25 + 0.5*np.cos(x0[2]))*np.sin(x0[2])*x0[3]) / (3.5 + np.cos(x0[2])) - np.sin(x0[2])*x0[1]) / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2]))),
                                (((1.25 + 0.5*np.cos(x0[2]))*(-4.9*np.sin(x0[0] + x0[2]) - 0.5*(x0[3]**2)*np.cos(x0[2]) - np.cos(x0[2])*x0[1]*x0[3]) - 0.5*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])*np.sin(x0[2])) / (3.5 + np.cos(x0[2])) + 4.9*np.sin(x0[0] + x0[2]) + (((1.25 + 0.5*np.cos(x0[2]))*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])) / ((3.5 + np.cos(x0[2]))**2))*np.sin(x0[2]) - 0.5*(x0[1]**2)*np.cos(x0[2])) / (1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2]))) - ((((1.25 + 0.5*np.cos(x0[2]))*(4.9*np.cos(x0[0] + x0[2]) + 14.7*np.cos(x0[0]) - 0.5*(x0[3]**2)*np.sin(x0[2]) - np.sin(x0[2])*x0[1]*x0[3])) / (3.5 + np.cos(x0[2])) - 4.9*np.cos(x0[0] + x0[2]) - 0.5*(x0[1]**2)*np.sin(x0[2])) / ((1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2])))**2))*(((-((1.25 + 0.5*np.cos(x0[2]))**2)) / ((3.5 + np.cos(x0[2]))**2))*np.sin(x0[2]) + (0.5*(2.5 + np.cos(x0[2]))*np.sin(x0[2])) / (3.5 + np.cos(x0[2]))),
                                ((1.25 + 0.5*np.cos(x0[2]))*(-np.sin(x0[2])*x0[1] - np.sin(x0[2])*x0[3])) / ((1.25 + (-((1.25 + 0.5*np.cos(x0[2]))**2)) / (3.5 + np.cos(x0[2])))*(3.5 + np.cos(x0[2])))]
                                    ], dtype=np.float32)
        pre_dot = np.dot(led_double_pendulum, identity)
        return pre_dot

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


    # def f_x (self, dyn, f, dt, x0, action):
    #     #change to get one x sample at a time
    #     x, v = self.RungeKutta(dyn, f, dt, x0, action)
    #     #x_noaction = self.RungeKutta(dyn, f, dt, x0, np.array([0.0, 0.0, 0.0]))
    #     return x, v
    
    def f_x (self, dyn, f, dt, x0, action):
        #change to get one x sample at a time
        x = f(x0, dyn, dt, action)
        #x_noaction = self.RungeKutta(dyn, f, dt, x0, np.array([0.0, 0.0, 0.0]))
        return x, 0
    
    def precal_le(self, s):
        
        index = self.get_spatial.query(s)[1]
        le = self.precal_le_[index]

        # print ('state:', s)
        # print ('nearest point:', self.precal_points[index])
        # print ('le at point:', le)
        return le


    def le_reward (self, s , action):
        T = self.T
        x = np.empty(shape=(len(s),T), dtype=np.float32)
        v1_prime = np.empty(shape=(len(s),T), dtype=np.float32)
        v2_prime = np.empty(shape=(len(s),T), dtype=np.float32)
        v3_prime = np.empty(shape=(len(s),T), dtype=np.float32)
        v4_prime = np.empty(shape=(len(s),T), dtype=np.float32)
        v1 = np.array([1, 0, 0, 0], dtype=np.float32)
        v2 = np.array([0, 1, 0, 0], dtype=np.float32)
        v3 = np.array([0, 0, 1, 0], dtype=np.float32)
        v4 = np.array([0, 0, 0, 1], dtype=np.float32)
        x[:, 0] = s
        v1_prime[:, 0] = v1
        v2_prime[:, 0] = v2
        v3_prime[:, 0] = v3
        v4_prime[:, 0] = v4
        le = np.array([0,0,0,0], dtype=np.float32)
        for i in range(1,T):

            x[:, i] = self.RungeKutta(self.dyn, self.f, self.dt, x[:, i-1])

            v1_prime[:, i] = self.RungeKutta_linearized(self.dyn, self.linearized_f, self.dt, v1_prime[:, i-1], x[:, i-1])
            v2_prime[:, i] = self.RungeKutta_linearized(self.dyn, self.linearized_f, self.dt, v2_prime[:, i-1], x[:, i-1])
            v3_prime[:, i] = self.RungeKutta_linearized(self.dyn, self.linearized_f, self.dt, v3_prime[:, i-1], x[:, i-1])
            v4_prime[:, i] = self.RungeKutta_linearized(self.dyn, self.linearized_f, self.dt, v4_prime[:, i-1], x[:, i-1])
    #         print (v4_prime[:, i])
    #         input()

            norm1 = np.linalg.norm(v1_prime[:, i])
            v1_prime[:, i] = v1_prime[:, i]/norm1

            GSC1 = np.dot(v1_prime[:, i], v2_prime[:, i])

            v2_prime[:, i] = v2_prime[:, i] - GSC1*v1_prime[:, i]
            norm2 = np.linalg.norm(v2_prime[:, i])
            v2_prime[:, i] = v2_prime[:, i]/norm2

            GSC2 = np.dot(v3_prime[:, i], v1_prime[:, i])
            GSC3 = np.dot(v3_prime[:, i], v2_prime[:, i])

            v3_prime[:, i] = v3_prime[:, i] - GSC2*v1_prime[:, i] - GSC3*v2_prime[:, i]
            norm3 = np.linalg.norm(v3_prime[:, i])
            v3_prime[:, i] = v3_prime[:, i]/norm3

            GSC4 = np.dot(v4_prime[:, i], v1_prime[:, i])
            GSC5 = np.dot(v4_prime[:, i], v2_prime[:, i])
            GSC6 = np.dot(v4_prime[:, i], v3_prime[:, i])

            v4_prime[:, i] = v4_prime[:, i] - GSC4*v1_prime[:, i] - GSC5*v2_prime[:, i] - GSC6*v3_prime[:, i]
            norm4 = np.linalg.norm(v4_prime[:, i])

            v4_prime[:, i] = v4_prime[:, i]/norm4

            le = le + np.log2(np.array([norm1,norm2,norm3,norm4]))

    #         if ( i % 100 == 0 ):
    #             print ('log2:', np.log2(np.array([norm1,norm2,norm3]))/(i*dt))
    #             print ('cum:', cum/(i*dt))
        le = le/(T*self.dt)

        return max(le)
    
    def step(self, u):
        x = self.state  # th := theta
        # x[0] = x[0]+np.pi/2
        dyn = self.dyn
        f = self.double_pendulum
        dt = self.dt
        # u = np.clip(u, -self.max_torque, self.max_torque)
        newx, v = self.f_x ( dyn, f, dt, x, u)

        # newx[0] = np.clip(newx[0], self.low_env_range[0], self.high_env_range[0])
        # newx[2] = np.clip(newx[2], self.low_env_range[2], self.high_env_range[2])
        # newx[1] = np.clip(newx[1], self.low_env_range[1], self.high_env_range[1])
        # newx[3] = np.clip(newx[3], self.low_env_range[3], self.high_env_range[3])
        if bool((newx >= self.high_env_range).any() or (newx <= self.low_env_range).any()):
            # print ('here')
            self.cost = -10
        else:
            self.cost = max(self.precal_le(newx))
        # self.cost = max(self.precal_le(newx))
        terminated = False
        if self.infinite == False:
            terminated = (newx <= self.high_env_range).all() and (newx >= self.low_env_range).all()
            if terminated == True:
                self.cost = -5

        self.state = newx

        return self._get_obs(), self.cost, terminated, {'trajectory': x, 'action': u}

    def reset(self):
        #put even closer to reward
        # high = np.array([np.pi, 1.0])
        # self.state = self.np_random.uniform(low=-high, high=high)
        if self.close_to_goal == True:
            self.state = self.np_random.uniform(low=self.goal_state-0.5, high=self.goal_state+0.5)
        else:
            self.state = self.np_random.uniform(low=self.low_env_range, high=self.high_env_range)
        #self.collected_states = self.collected_states.append(self.state)
        self.last_u = None
        self._render_reset()
        return self._get_obs()


    def _render_reset(self):
        self.ax = self.fig.add_subplot()
        self.ax.scatter(0,0, s=100, color="red")

    def _get_obs(self):
        theta1, thetadot1, theta2, thetadot2 = self.state
        # print ( self.state)
        #self.collected_states.append(self.state)
        # return np.array([np.cos(theta1), np.sin(theta1), thetadot1, 
        #                    np.cos(theta2), np.sin(theta2), thetadot2 ], dtype=np.float32)
        return np.array([np.cos(theta1), np.sin(theta1), thetadot1, 
                    np.cos(theta2), np.sin(theta2), thetadot2 ], dtype=np.float32)

    def render(self, mode="human"):
        x = self.state
        u = self.action_u
        x_noaction_local = self.x_noaction

        print ("no action:", x_noaction_local)
        print ("with action:", x)
        print ('action:', u)
        for i, m, k in [(x, 'o', 'green'), (x_noaction_local, '^', 'blue')]:
            self.ax.scatter3D(i[0], i[1],s=10,c=k,marker=m, alpha=0.5)
        #self.ax.scatter3D(x_noaction_local[0], x_noaction_local[1], x_noaction_local[2], s=10, c='blue', alpha=0.5)
        plt.title('pendulum attractor')
        plt.draw()
        #plt.show(block=False)
        #self.collected_states = list()
        plt.savefig('pendulum_ppo_2d.png')

#empowerment guy