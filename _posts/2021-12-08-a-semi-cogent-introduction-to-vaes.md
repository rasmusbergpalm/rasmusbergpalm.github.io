---
layout: default
title: "A semi-cogent introduction to VAEs"
---

# A semi-cogent introduction to VAEs

There's been many tutorials and great introductions written on Variational Auto-Encoders (VAEs). 
One thing I found frustrating about VAEs when I first started learning about them was that the tutorials would usually pull a bait and switch.
This is my attempt at an introduction to VAEs, starting from pretty much scratch and trying to present a cogent argument throughout.  
 
So what is the bait and switch? First a tutorial will start by saying we want to make a generative model, so we should maximize $\log p_\theta(x)$. So far so good.
But then they would say, without any argument as to why, well actually, let's instead look at a KL divergence between the true (unknown) posterior $p(z|x)$ and some weird encoder distribution $q(z|x)$, and after some algebra, by pure magic, that would actually turn out to be a lower bound on $\log p_\theta(x)$, the thing we wanted to optimize! 
That surely is impressive, but there was no argument *why* anyone would look at this KL divergence in the first place. From just reading the introductions in those tutorials, I could never come up with looking at that KL divergence myself. It wasn't logical to me. 

Now I realize, that there's a rich history of considering this KL divergence in variational inference (VI), for its own sake, and that it was known that it lead to a lower bound on $\log p(x)$. So the argument could simply be, "we happen to know this lower bound on $\log p(x)$ from variational inference, go read up on that". 

That's fair, but I would have loved a single cogent argument starting from "let's build a generative model" straight to the lower bound (ELBO), and amortized VI. So I spent some time coming up with my best stab at a semi-cogent argument, from nothing all the way to the ELBO/amortized VI, through importance sampling. The cool part about doing it this way is you realize all the branches and choices along the way, which lead me to the somewhat surprising result that you can actually train a semi-decent model of MNIST by just sampling from the prior, no encoder or VI needed at all.

<img src="https://drive.google.com/uc?id=19gxET4Bqv8O9HD0HpbY5Y8U-KRQMYUNn" />
<br />
<small> MNIST samples from a "VAE", that is not variational and does not auto-encode, it just samples from the prior. </small>

The tutorial itself is in colab, and the format is that of a small seminar, but it should be readable on its own. I hope you enjoy it.

https://colab.research.google.com/drive/1NiqTSUr_YsBHSiizuSZUw2j9p-vGTbBA?usp=sharing