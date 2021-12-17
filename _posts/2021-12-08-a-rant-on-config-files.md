---
layout: post
title: "A rant on config files"
---

# A rant on config files

I see a lot of machine learning projects, from top groups and newcomers alike, that use configuration files to store hyper-parameters, architecture details and experimental setups. I don't like it, and I'll try to explain to you why I think it's a bad idea.

## You are inventing your own incomplete, outdated, leaky-abstraction API, with no IDE support, and no docs.
Your config file starts simple, just a learning rate, and maybe the number of hidden units.
```
lr: 1e-3
n_hidden: 128
```
Surely, this is not too bad. But then you want to experiment with different number of hidden units in the layers, so you get creative.
```
lr: 1e-3
n_hidden: [128, 256]
```
Ah, yea. Not too bad. But now you want to try some different activation functions.
```
lr: 1e-3
layers: 
    - n_hidden: 128
      act_fn: relu
    - n_hidden: 256
      act_fn: elu  
```
You see where this is going. Before long you are re-inventing the entire API of Keras or pytorch in yaml, **with no formal specification**. The only specification of your newly invented format is in how your code reads it.

**With no documentation**. What options does your config file support? Can I use leaky relu activation functions? What about swish, tanh, sigm, softplus? is that `leaky_relu`? How do i set the alpha parameter of the leaky relu? Who knows. The only way to find out is to read your code.

**With no IDE support**. Maybe I find an interesting config value, and think, huh, I'd love to see where that is used. Too bad. My IDE does not understand your config file. Or the other way around; I'm reading your code and comes upon `config["loss"]`, and I'd love to know what that is. Cmd-click gets me a bunch of yaml config parsing. Again, the only way to see where a config value is used and vice versa is by reading your code, your config file, and your config parsing logic simultaneously, which is a pain with only two eyes and one brain.

**It's bound to be incomplete**. What about initializations of the layers? So after reading your code I realize that I can't set that in the config file, OK, so now I have to 1) expand your config file format and add parsing logic or 2) modify it in code. So now I have to modify the same layer in two places.

**It's bound to be outdated**. You thought you wanted to have different activation functions per layer, but in the end you ended up using leaky-relu with different alpha parameters instead. And you didn't have time to change the config file, so you just hardcoded it (it was 4 hours till conference deadline, you're forgiven). So, now you have a config file, which may use the `n_hidden` from the config file, but ignores the activation function. Do you have any idea how hard it is to infer that from just looking at your config file? You're right, it's impossible, so I have to read your code anyways, and verify that indeed all your config values are used for what I'd imagine they are.  

**I have to read your code anyways!** See how I have to read your code anyways? You might have meant for the config file to make my/your life easier. That has not been the case. I still have to read your code, but now you've just created another layer of abstraction I have to understand.

**What's the point?** Where does the idea of config files come from? They come from software development where they allow **users** to configure software without having to modify and re-compile it. That is great and fine. But I'd argue that in 99% of cases your users are other machine learning practitioners, or, most likely, your future self, which very likely want to change something that is not in the config file. And even if it was in the config file, why is it easier to change a value in a yaml file than in a python file, where you have IDE support and docs? If you are legit releasing GPT-3 to a worldwide audience, and it's impossible for users to modify the model anyways (due to re-training costs), but they can tune a few value, e.g. the temperature, and promt, and you expect a lot of normal people as users, and you write extensive docs and tests of your config file format, then feel free to ignore everything I've said. Great job! But if you're "just" creating some code for your next research project, you do not need a config file. And I, as a future reader of your code definitely do not want one. 

**Config values as global variables and if soup**. On top of everything I've said there's a specific anti-pattern which I've seen more often with config files than without. To be fair you can also do this without a config file, (which you shouldn't), but the lack of IDE support makes this especially bad if you do use a config file.

Try reading the following code:

```python
if config["data"] == "mnist":
    x,y = load_mnist()
elif config["data"] == "cifar":
    x,y = load_cifar()

if config["use_encoder"]:
    self.encoder = Encoder(config["encoder"])
    z = self.encoder(x).sample()
else:
    z = self.prior.sample()
    
if config["loss"] == "elbo":
    loss = elbo(decoder(z), x, config["beta"])
elif config["loss"] == "iwae":
    loss = iwae(decoder(z), x, config["iwae_params"])
    
if config["regularizer"]:
    ... #etc
```

What is this code actually doing at any one point? It's impossible to figure out without simultaneously reading and live-parsing config files, and because there's no IDE support I can't just cmd-click at the variables and see what they are.

**A very simple fix**. The first step, which helps immensely is to just take your yaml config file and turn it into python. Just a bunch of variables in a file named config.py. In an instant you have IDE support! 
Yay! And you can actually use functions and objects as config values e.g instead of `act_fn: 'relu'` and associated logic to turn that string into an activation function, you can just say `act_fn = t.nn.ReLU()`. 
Next inline all those variables. OK, that last bit was a bit tongue-in-cheek, but try it. 
Which variables do you really need in your config.py? Are they making your life easier? Are they making my life easier? What about once you've forgotten where they're all used?

**Towards better code**. First off, realize that during development your research code will change too rapidly for any config file to make sense. 
Just hardcode the values and **use version control**. I'll get into the details of my setup in a later post. 
Feel free to duplicate code if it makes it easier. 
I know, I know, code duplication is a major anti-pattern. Yes. 
For production software. Are you writing production software? 
Duplicate it now, and if you legit end up using both copies, with minor differences, make an abstraction. 
My guess is that in 95% of the cases, you'll pick one, or none, and never have to worry about it, because you needed to change everything anyways.
Once your model and paper is done, and the reviewers are happy, and the paper is published, now is the time to make your code nice, so it can be published. 
So what should it look like? 
The important part of code for papers is that you can reproduce all the experiments. 
So make a single python file per experiment which wires up your model in such a way that it can reproduce the experiment. 
You must be able to run this one file, with no command line parameters, no config files, and no modifications, and it must reproduce your experiment. 
Something like this. 
```
model/
 - my_ultra_deep_vae.py
 - crazy_loss.py
 - all your model files here, structured however you like
experiments/
 - experiment_1.py
 - experiment_2.py  
README.md explains which file to run to reproduce which experiment.
```