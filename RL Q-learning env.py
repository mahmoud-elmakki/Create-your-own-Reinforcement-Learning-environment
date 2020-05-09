import numpy as np
#from PIL import Image
#import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
#import time

style.use("ggplot")

SIZE = 10
HM_EPISODES = 1000
STEPS = 200
MOVE_PANELTY = 1
ENEMY_PANELTY = 100
FOOD_REWARD = 25 

epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 100

start_q_table = None

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}

class Blob:
    
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
        
    # returns a string   
    def __str__(self):
        return f"{self.x}, {self.y}"
    
    # allows us to subtract a blob from another blob
    def __sub__(self, other):
        return (self.x - other.x, self.y - other.y)
        
    def action(self, choice):
        if choice == 0:
            self.move(x=1, y=1)
        if choice == 1:
            self.move(x=-1, y=-1)
        if choice == 2:
            self.move(x=-1, y=1)
        if choice == 3:
            self.move(x=1, y=-1)
            
    def move(self, x = False, y = False):
        # if three is no values passed, move randomly
        if not x:
            self.x = np.random.randint(-1, 2)
        else:
            self.x += x
            
        if not y:
            self.y = np.random.randint(-1, 2)
        else:
            self.y += y
            
        # when it hits the walls
        if self.x < 0:
            self.x = 0
        if self.x > SIZE - 1:
            self.x = SIZE - 1

        if self.y < 0:
            self.y = 0
        if self.y > SIZE - 1:
            self.y = SIZE - 1

if start_q_table is None:
    q_table = {}
    # the observation is two tupels
    # one for the deference between the player and the food
    # one for the deference between the player and the enemy
    
    # put all possible states in the Q table
    for xpf in range(-SIZE + 1, SIZE):
        for ypf in range(-SIZE + 1, SIZE):
            for xpe in range(-SIZE + 1, SIZE):
                for ype in range(-SIZE + 1, SIZE):
                    q_table[(xpf, ypf), (xpe, ype)] = [np.random.uniform(-5, 0) for i in range(4)]
                    
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []      
for episode in range(HM_EPISODES):      
    player = Blob()
    food = Blob()
    enemy = Blob()
    
    if episode % SHOW_EVERY == 0:
        print(f"on # {episode} , epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
        
    episode_reward = 0
    for i in range(STEPS):
        obs = (player - food, player - enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
            
        player.action(action)
        
        # enemt.move()
        # food.move()
        
        if player.x == enemy.x and player.y == enemy.y:
            reward = - ENEMY_PANELTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = - MOVE_PANELTY
      
        new_obs = (player - food, player - enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]
        
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == - ENEMY_PANELTY:
            new_q = - ENEMY_PANELTY
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            
        q_table[obs][action] = new_q   
        
        if show:
            env  = np.zeros((SIZE, SIZE, 3), dtype = np.uint8)
            env[food.y][food.x] = d[FOOD_N]
            env[player.y][food.x] = d[PLAYER_N]       
            env[enemy.y][food.x] = d[ENEMY_N]
            
#            img = Image.fromarray(env, "RGB")
#            img = img.resize(300, 300)
            
#            cv2.imshow("", np.array(img))
            
#            if reward == FOOD_REWARD or reward == -ENEMY_PANELTY:
#                if cv2.waitkey(500) and 0xff == ord("q"):
#                   break
#            else:
#                if cv2.waitkey(1) and 0xff == ord("q"):
#                    break
        
        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PANELTY:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY       
            
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY} ma")           
plt.xlabel("episode #")      
plt.show()          
            
#with open(f"qtable-{int(time.time())}.pickle", "wb") as f:\
#    pickle.dump(q_table, f)      
        
        
        
        
        
        
        
        
        