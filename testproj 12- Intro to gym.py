import gym
#Importing openAI gym, helps us work with cool enviornments and work with reinforcement models

env = gym.make('FrozenLake-v1', render_mode = "rgb_array")             #Going to use the frozen lake enviornment
#The frozen lake enviornment is a basic game where we try to go through a frozen lake without falling through the ice
#There are 16 states(one for each square)
#There are 4 actions, (up, down, left, right)
#There are 4 different types of blocks, (Frozen, hole, start, goal)


print(env.observation_space.n)              #Print number of states, aka 16 for this env
print(env.action_space.n)                   #Print number of actions per state, aka 4

env.reset()
#Resets enviornment to default state, aka 0 for this enviornment

action = env.action_space.sample()
#Get a random action from the enviornment, get a random sample from the action space. 

new_state, reward, done, truncated, info = env.step(action)
#Take the given random action in the environemnt. It will return 5 values, first the state it is now in, 
#Second the reward received for the given action, 
#Third tells us if we won or lost the game, ie. if we are no longer in a valid state in the enviornment. So if the done variable is true, we need to reset the state using the command above
#These are the main values we will use in this example, for more info just read the documentation, gives in dept explanation for each return value. 

