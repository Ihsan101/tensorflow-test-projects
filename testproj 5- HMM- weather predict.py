import tensorflow_probability as tfp
import tensorflow as tf
import time 

#Creating a weather prediction model, using a Hidden Markov Model
#Cold days are encoded by a 0 and hot days encoded with a 1-----------------1
#First day in our sequence has an 80% chance to be cold---------------------2
#A cold day has a 30% chance of being followed by a hot day-----------------3
#a hot day has a 20% chance of being followed by a cold day ----------------4
#On each day the temperature is normally distributed with mean and standard deviation 0 and 5 on a cold day and mean and standard deviation 35 and 10 on a hot day.--------------------5


tfd = tfp.distributions                                         #Making a shortcut so we dont need to say tfp.distributions.Categorical every single time.

initial_distribution = tfd.Categorical(probs=[0.8, 0.2])        #Creating intiial distribution, containing first probablity distribution. As we can see, the cold days are in the 0 index and hot in 1 Refering to point 1
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3],    #0 index is cold day, so a cold day has 0.3 chance of hot day, so 0.7 chance of being followed by a cold day. List is created accordingly. Refering to point 2
                                                 [0.2, 0.8]])   #1 index is hot day, and has a 0.2 chance of being followed by a cold day, so its in index 0 for index 1 in the list. Refering to point 3
                                                                #THESE ARE THE TRANSITIONAL PROBABILITIES

observation_distribution = tfd.Normal(loc=[0., 35.], scale=[5., 10.])       #loc gives the mean and scale gives the standard deviation of cold and hot day in that order. Do not forget the dot, these values need to be float values. Refering to point 5

#In this case scenario we used cold to be encoded by 0 and hot to be encoded by 1. It doesnt matter what order we put them, as long as we make sure to use it throughout the document properly.

model = tfd.HiddenMarkovModel(
    initial_distribution = initial_distribution,
    transition_distribution = transition_distribution,
    observation_distribution = observation_distribution,
    num_steps = 7
)

#Number of steps is the number of days to predict. More steps for more days, more states, so more observations.

mean = model.mean()
#We cant just get the value immediately, we need to evalute a part of the graph, to get a partially defined tensor 

#To create a session we need to specify compat.v1.Session() instead of just tf.Session() in TensorFlow versions 1.x
with tf.compat.v1.Session() as sess:
    print(mean.numpy())
time.sleep(5)

#As time runs on, since this is just a probability model you cant really guess huge number of days with this model, as it will increasingly get more innacurate. 