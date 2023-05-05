import gym
from gym import spaces
import random
from torch.utils.data import DataLoader
from datasets import *
from sklearn.metrics import mean_squared_error

class FractalEnv(gym.Env):
    #add the metadata attribute to your class
    def __init__(self):

        dataloader = DataLoader(FractalDataset("./data/%s"),
            batch_size=12,
            shuffle=True,
            num_workers=8,
        )
        self.observation_space= gym.spaces.Box(low=0, high=100, shape=(1,))
         
        self.action_space= gym.spaces.Discrete(3)
        
        # current state
        self.state= random.randint(0,20)
        
        # rewards 
        self.reward=0
        self.findex=0
    
    def get_action_meanings(self, action):
        pass

    
    def step(self, action_tf): 
        data=next(dataloader)
        mse=mean_squared_error(data[0],action_tf)
        self.reward=1-mse
        return [data[1],self.reward]

        
        
    def render(self,action):
       pass
    def reset(self):
        #reset your environment
        self.state=random.randint(0,20)
        self.reward=0
        return self.state
    def close(self):
        # close the nevironment
        self.state=0
        self.reward=0