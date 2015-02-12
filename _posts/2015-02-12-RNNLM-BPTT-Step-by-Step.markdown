---
layout: post
title: Recurrent Neural Network Language Model BPTT Step by Step
date: 2015-02-12 19:15:38 
categories: jekyll update
---
**Recurrent Neural Network (RNN)** is known as to be able to capture longer history information compared to simple feedforward neural network language model.
This advantage is achieved by including a recurrent unit in the hidden layer. 
The output from the hidden layer in previous step will be the a part of input of the hidden layer in current step.
Then history information can be encoded to the hidden layer neuron for next step computation.    

Similar to feedforward network, we still use backpropagation alogrithm to optimize parameters in this network.
But the recurrent architecture makes the backpropagation procedure a little bit difficult for newbies to have a full understanding of mathematical details. 
The backpropagation alogrithm used in recurrent neural network is so called **backprogagation through time (BPTT)**, where we need to consider the timestamp. 
In this blog post, We intend to lay specail emphasis on mathematical details of BPTT alogrithm. 
Specifically, we will cover these questions:  

+  How to unfold a recurrent network generally?  
+  How to derivate only by chain rules?
+  How to understand backpropagation from **Error Propagation**?
+  What is the weight updating mechanism after unfolding? And why we need to do that?  

After reading this blog, I hope you can have a better understanding of RNNLM and BPTT alogrithm.  

###Structure of Recurrent Neural Network Language Model   

The network structure of RNNLM is shown in [Figure 1][f1].  
[![Figure 1](/images/rnn_structure.png "Network Structure of Recurrent Neural Network Language Model")](f1)  

+ Input Layer. \\(x(t)\\) is the input of time \\(t\\), a \\(V\\) by \\(1\\) vector. 
Only one component is \\(1\\), others are \\(0\\). 
The size of \\(x(t)\\) equals the size of vocabulary \\(V\\). 
In general situation, the size of \\(x(t)\\) and \\(y(t)\\) may not be the same.
+ Hidden Layer. \\(s(t)\\) is the hidden layer output of time \\(t\\), a \\(H\\) by \\(1\\) vector. 
\\(H\\) is the number of neurons in hidden layer.
+ Output Layer. \\(y(t)\\) is the output of time \\(t\\), a \\(V\\) by \\(1\\) vector. 
+ Input Layer. \\(s(t-1)\\) is the hidder layer output of time \\(t-1\\), a \\(H\\) by \\(1\\) vector. 

+ Input-to-Hidden Matrix \\(U\\), \\(H\\) by \\(V\\).
+ Hidden-to-Output Matrix  \\(W\\), \\(V\\) by \\(H\\). 
+ Previous-to-Current Matrix \\(R\\), \\(H\\) by \\(H\\). 
The red line in [Figure 1][f1] means that the output of hidden layer at time \\((t-1)\\) is used as the input of hidden layer at time \\(t\\).   

This network can be used to predict \\(y(t)\\), 
\begin{equation}
 y(t)=P(x\_{t+1}|x\_t,s(t-1)) 
\label{1}
\end{equation}
In a feedforward neural network language model, suppose we predict a word only by two previous words, then
\begin{equation}
 y(t)=P(x\_{t+1}|x\_t,x\_{t-1}) 
\label{2}
\end{equation}

Comaring equation $\eqref{1}$ and equation $\eqref{2}$, we also can tell the difference between feedforward netword and recurrent network since \\(s(t-1)\\) can contain longer history information not just limited to some local words.   

---

###Feedforward Propagation  
Usually, the non-linear transform function in hidden layer and output layer is **sigmoid function** $f$ and **softmax function** $g$ respectively. There are other options for $f$ and $g$, but the basic idea is same.
First, we look at function $f$,
$$ f(x) = \frac{1}{1+\mathrm{e}^{-x}} $$
And the derivation of $f(x)$ with respect to $x$ has an interesting property,
\begin{equation}
 f^\prime(x) = f(x) (1-f(x)) 
\label{3}
\end{equation}
We will use equation $\eqref{3}$ many times later.  The form of $g$ is,
\begin{equation}
g(c\_k) = \frac{\mathrm{e}^{c\_k}}{\sum\_{k}\mathrm{e}^{c\_k}}
\label{4}
\end{equation}  

\begin{equation}
 b\_j(t) = \sum\_{i=1}^{V}x\_i(t)\*u\_{ji} + \sum\_{h=1}^{H}s\_l(t-1)\*r\_{jl}
\label{5}
\end{equation}

\begin{equation}
s\_j(t) = f(b\_j(t))
\label{6}
\end{equation}

\begin{equation}
c\_k(t) = \sum\_{h=1}^{H}s\_h(t)\*w\_{kh}
\label{7}
\end{equation}
\begin{equation}
y\_k(t) = g(c\_k(t))
\label{8}
\end{equation}

---
###Backward Progagation  
1. Cost Function  
Support the correct output is $d(t)$ at time $t$. There is only one component in $d(t)$ is $1$, others are $0$.   
We want to make $y(t)$ close to $d(t)$. 
$d(t)$ and $y(t)$ are two probability distributions. 
So we can define a distance function between $d(t)$ and $y(t)$ by cross entropy criterion,
\begin{equation}
    L(t) = -\sum\_{k=1}^{V} d\_k(t)\log y\_k(t)
\label{9}
\end{equation}

---
###Backward Progagation Through Time
* Unfolding
* Weight Updating Mechanism  

###References
[f1]:/f1
