---
layout: default
---

# About me

I'm a machine learning researcher and engineer. I'm broadly interested in probabilistic machine learning, generative models, bayesian statistics, unsupervised and self-supervised learning and inductive priors.

I'm currently working at [twig.energy](https://www.twig.energy/), trying to remove some carbon from the power grid. Prior to that I did a PostDoc in the [REAL](https://real.itu.dk/) group at the IT University of Copenhagen working with [Sebastian Risi](http://sebastianrisi.com/). Before joining ITU I did my PhD at the [Cognitive Systems](https://www.compute.dtu.dk/english/research/research-sections/cogsys) group at the Technical University of Denmark with [Ole Winther](https://olewinther.github.io/). Prior to that I was a Staff Engineer and Machine Learning Team lead at [Tradeshift](https://tradeshift.com/).

> "What I cannot create, I do not understand." - Richard Feynman

## Publications

- **Rasmus Berg Palm**, Miguel González-Duque, Shyam Sudhakaran and Sebastian Risi. [Variational Neural Cellular Automata](https://arxiv.org/abs/2201.12360) ICLR 2022
- **Palm, Rasmus Berg**, Elias Najarro, and Sebastian Risi. [Testing the genomic bottleneck hypothesis in Hebbian meta-learning](https://arxiv.org/abs/2011.06811) NeurIPS 2020 Workshop on Pre-registration in Machine Learning. PMLR, 2021.
- González-Duque, Miguel, **Rasmus Berg Palm**, and Sebastian Risi. [Fast Game Content Adaptation Through Bayesian-based Player Modelling](https://arxiv.org/abs/2105.08484) arXiv preprint arXiv:2105.08484 (2021).
- Olesen, Thor V.A.N., Dennis T.T. Nguyen, **Rasmus Berg Palm** and Sebastian Risi. [Evolutionary Planning in Latent Space](https://arxiv.org/abs/2011.11293) International Conference on the Applications of Evolutionary Computation. 2021.
- **Palm, Rasmus Berg** and Pola Schwöbel. [Justitia ex Machina: The Case for Automating Morals](https://thegradient.pub/justitia-ex-machina/). The Gradient, 2021.
- Grbic, Djordje, **Rasmus Berg Palm**, Elias Najarro, Claire Glanois and Sebastian Risi. [EvoCraft: A New Challenge for Open-Endedness](https://arxiv.org/abs/2012.04751) EvoApplications. 2021.
- González-Duque, Miguel, **Rasmus Berg Palm**, David Ha and Sebastian Risi. [Finding Game Levels with the Right Difficulty in a Few Trials through Intelligent Trial-and-Error](https://arxiv.org/abs/2005.07677) 2020 IEEE Conference on Games (CoG). IEEE, 2020.
- **Palm, Rasmus Berg**, Florian Laws, and Ole Winther. [Attend, copy, parse end-to-end information extraction from documents](https://arxiv.org/abs/1812.07248) 2019 International Conference on Document Analysis and Recognition (ICDAR). IEEE, 2019.
- **Palm, Rasmus Berg**. [End-to-end information extraction from business documents](https://orbit.dtu.dk/en/publications/end-to-end-information-extraction-from-business-documents) English, PhD thesis (2018).
- **Palm, Rasmus Berg**, Ulrich Paquet, and Ole Winther. [Recurrent Relational Networks](https://arxiv.org/abs/1711.08028) Proceedings of the 32nd International Conference on Neural Information Processing Systems. 2018.
- **Palm, Rasmus Berg**, Ole Winther, and Florian Laws. [Cloudscan - a configuration-free invoice analysis system using recurrent neural networks](https://arxiv.org/abs/1708.07403) 2017 14th IAPR International Conference on Document Analysis and Recognition (ICDAR). Vol. 1. IEEE, 2017.
- **Palm, Rasmus Berg**, Dirk Hovy, Florian Laws and Ole Winther. [End-to-End Information Extraction without Token-Level Supervision](https://arxiv.org/abs/1707.04913) Proceedings of the Workshop on Speech-Centric Natural Language Processing. 2017.
- **Palm, Rasmus Berg**. [Prediction as a candidate for learning deep hierarchical models of data](https://www2.imm.dtu.dk/pubdb/pubs/6284-full.html) Master thesis - Technical University of Denmark (2012).

## Software

I often create software libraries and tools and I love publishing them as open source. It lets me write clean code, think about abstractions and interfaces and ship working software to users, which I enjoy.

- [nanograd](https://github.com/rasmusbergpalm/nanograd) - The simplest and smallest possible library for autograd. Great for teaching.
- [shapeguard](https://github.com/rasmusbergpalm/shapeguard) - A tiny library, which allows you to very succinctly assert the expected shapes of tensors in a dynamic, einsum inspired way. A great tool for avoiding bugs.
- [EvoStrat](https://github.com/rasmusbergpalm/evostrat) - A library that makes Evolutionary Strategies (ES) simple to use. It has a flexible and natural interface for ES that cleanly separates the environment, the reinforcement learning agent, the population distribution and the optimization, which allows you to use the standard pytorch `nn.Modules` for policy networks and `torch.optim` optimizers for optimization.
- [pymc3-quap](https://github.com/rasmusbergpalm/pymc3-quap) A quadratic approximation package for PyMC3.
- [pytorch-lgssm](https://github.com/rasmusbergpalm/pytorch-lgssm) A clean Linear Gaussian State Space Model for pytorch, which supports sampling and inference using the Kalman filtering algorithm.
- [Blayze](https://github.com/Tradeshift/blayze) A very fast and efficient *Bayesian* Naive Bayes classifier which perfectly incorporates new information in an online learning setting and support both Gaussian and Categorical/Multinomial features.
- [DeepLearnToolbox](https://github.com/rasmusbergpalm/DeepLearnToolbox) - A now deprecated Matlab toolbox for Deep Learning. I was unwise enough to do my Masters on Deep Learning in Matlab, which didn't have any support for it, so I made a library for MLPs, CNNs, DBN, etc, implementing everything from the ground up, including the gradients, backprop, etc. It became quite popular. I quickly moved on to Python though (yay autograd!), and deprecated it.

## Teaching

I thoroughly enjoy teaching. It challenges me to understand things at a much deeper level, and I find it very rewarding.

- [Advanced Applied Statistics and Multivariate Calculus](https://learnit.itu.dk/local/coursebase/view.php?ciid=789), 2021
- [Modern AI](https://learnit.itu.dk/local/coursebase/view.php?ciid=749), 2020 and 2021

