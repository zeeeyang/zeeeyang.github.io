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
But the recurrent architecture makes the backpropagation procedure a little bit difficult for *newbies* to have a full understanding of mathematical details. 
The backpropagation alogrithm used in recurrent neural network is so called **backprogagation through time (BPTT)**, where we need to consider the timestamp. 
In this blog post, We intend to lay specail emphasis on mathematical details of BPTT alogrithm. 
Specifically, we will cover these questions:  

+  How to unfold a recurrent network generally?  
+  How to calculate derivatives only by chain rules?
+  How to understand backpropagation from **Error Propagation**?
+  What is the weight updating mechanism after unfolding? And why we need to do that?  

After reading this blog, I hope you can have a better understanding of RNNLM and BPTT alogrithm.  

##Structure of Recurrent Neural Network Language Model   

The network structure of RNNLM is shown in Figure 1.   
![Figure 1](/images/rnn_structure.png "Network Structure of Recurrent Neural Network Language Model") 

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
The red line in Figure 1 means that the output of hidden layer at time \\((t-1)\\) is used as the input of hidden layer at time \\(t\\).   

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

##Feedforward Propagation  
Usually, the non-linear transform function in hidden layer and output layer is **sigmoid function** $f$ and **softmax function** $g$ respectively. There are other options for $f$ and $g$, but the basic idea is the same.
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
 b\_j(t) = \sum\_{i=1}^{V}x\_i(t)\*u\_{ji}(t) + \sum\_{h=1}^{H}s\_l(t-1)\*r\_{jl}(t)
\label{5}
\end{equation}

\begin{equation}
s\_j(t) = f(b\_j(t))
\label{6}
\end{equation}

\begin{equation}
c\_k(t) = \sum\_{h=1}^{H}s\_h(t)\*w\_{kh}(t)
\label{7}
\end{equation}
\begin{equation}
y\_k(t) = g(c\_k(t))
\label{8}
\end{equation}

---
##Backward Progagation  
### Cost Function  
What's the goal of training? We want to make the network output as close as possible to the corrent output.   
Support the correct output is $d(t)$ at time $t$. 
There is only one component in $d(t)$ is $1$, others are $0$. 
We want to minimize the distance between  $y(t)$ and $d(t)$.   
Since $d(t)$ and $y(t)$ are two probability distributions, we can define a distance function between $d(t)$ and $y(t)$ by cross entropy criterion,
\begin{equation}
    L(t) = -\sum\_{k=1}^{V} d\_k(t)\log y\_k(t)
\label{9}
\end{equation}
$U$,$R$,$W$ are parameters in RNNLM. 
We need to use training data to optimize these parameters.
Commonly, **Stochastic Gradient Descent** can find good enough parameters in a multi-layer network, no matter whether the cost function is a convex function or not.
Instead of using gradient descent, the author **Tomas Mikolov** employs stochastic gradient ascent to learn parameters. 
We follow his way in this post.
\begin{equation}
    L(t) = \sum\_{k=1}^{V} d\_k(t)\log y\_k(t)
\label{10}
\end{equation}

### Updating $W$  
First, we consider the parameter $W$. 
Take one element $w\_{kh}(t)$ in $W$ for example, in order to update $w\_{kh}(t)$,  we need to caculate the partial derivative $\frac{\partial{L(t)}}{\partial{w\_{kh}(t)}}$.
After that we can update $W$,
\begin{equation}
w\_{kh}(t+1) = w\_{kh}(t) + \alpha\*\frac{\partial{L(t)}}{\partial{w\_{kh}(t)}}-\beta\*w\_{kh}(t)
\label{11}
\end{equation}
where $\alpha$ is the learning rate and $\beta$ is the regularization parameter.  
$L(t)$ is a function of $w\_{kh}(t)$. However, directly calculating derivatives is difficult. 
Here, we use a common technique so called **chain rule** to calculate derivatives through the network. 
The chain from $w\_{kh}(t)$ to $L(t)$ is shown in Figure 2.
![Figure 2](/images/w_chain.png "Chain of W") 
According to the chain, we can calculate $\frac{\partial{L(t)}}{\partial{w\_{kh}(t)}}$ by 
\begin{equation}
  \frac{\partial{L(t)}}{\partial{w\_{kh}(t)}} = 
  \frac{\partial{L(t)}}{\partial{c\_{k}(t)}} \* \frac{\partial{c\_{k}(t)}}{\partial{w\_{kh}(t)}}
  =(\sum\_{o=1}^{V}{\frac{\partial{L(t)}}{\partial{y\_o(t)}}\* \frac{\partial{y\_o(t)}}{\partial{c\_k(t)}}}) \* \frac{\partial{c\_{k}(t)}}{\partial{w\_{kh}(t)}}
  \label{12}
\end{equation}
There are three items here, $\frac{\partial{L(t)}}{\partial{y\_o(t)}}$,$\frac{\partial{y\_o(t)}}{\partial{c\_k(t)}}$,$\frac{\partial{c\_{k}(t)}}{\partial{w\_{kh}}(t)}$. We will handle them one by one **from backward to forward**.  
(1) $\frac{\partial{L(t)}}{\partial{y\_o(t)}}$  
According to equation $\eqref{10}$, 
\begin{equation}
\frac{\partial{L(t)}}{\partial{y\_o(t)}} = \frac{d\_o(t)}{y\_o(t)}  
\label{13}
\end{equation}
(2) $\frac{\partial{y\_o(t)}}{\partial{c\_k(t)}}$  
According to equation $\eqref{4}$ and $\eqref{8}$,
\begin{equation}
\frac{\partial{y\_o(t)}}{\partial{c\_k(t)}} = \frac{\partial{g(c\_o(t))}}{\partial{c\_k(t)}} = y\_o(t)\*\delta\_{o,k} - y\_o(t)*y\_k(t)
\label{14}
\end{equation}  

Combining $\eqref{13}$ and $\eqref{14}$, we can achieve
\begin{equation}
\frac{\partial{L(t)}}{\partial{c\_{k}(t)}} = \sum\_{o=1}^{V}{\frac{\partial{L(t)}}{\partial{y\_o(t)}}\* \frac{\partial{y\_o(t)}}{\partial{c\_k(t)}}}
= \sum\_{o=1}^{V}{\frac{d\_o(t)}{y\_o(t)} * (y\_o(t)\*\delta\_{o,k} - y\_o(t)*y\_k(t))} = \sum\_{o=1}^{V}{d\_o(t) * (\delta\_{o,k} - y\_k(t))} = d\_k(t) - y\_k(t)\*\sum\_{o=1}^{V}{d\_o(t)} = d\_k(t) - y\_k(t)
\label{15}
\end{equation}
where $\delta\_{o,k}=1$ when $o=k$, otherwise $\delta\_{o,k}=0$.
$\sum\_{o=1}^{V}{d\_o(t)} = 1$ since only one element in $d(t)$ is $1$, others are $0$.   
Equation $\eqref{15}$ is very important, so we give it a short name $e\_k(t)$ for later use. 

\begin{equation}
e\_k(t) = \frac{\partial{L(t)}}{\partial{c\_{k}(t)}} = d\_k(t) - y\_k(t)
\label{16}
\end{equation}

$e\_k(t)$ has a beautiful form. It means the **error** or **difference** between predicted output $y\_k(t)$ and correct answer $d\_k(t)$.  

(3) $\frac{\partial{c\_{k}(t)}}{\partial{w\_{kh}}(t)}$  
This term is very easy. By using equation $\eqref{7}$, we can get 
\begin{equation}
\frac{\partial{c\_{k}(t)}}{\partial{w\_{kh}}(t)} = s\_h(t) 
\label{17}
\end{equation}

Now, we replace equation $\eqref{16}$ and equation $\eqref{17}$ into equation $\eqref{12}$, we can get what we want.

\begin{equation}
\frac{\partial{L(t)}}{\partial{w\_{kh}(t)}} = e\_k(t) * s\_h(t)
\label{18}
\end{equation}

Wow...simple and beautiful, isn't it?  
So the updating rule for $w\_{kh}$ is 
\begin{equation}
w\_{kh}(t+1) = w\_{kh}(t) + \alpha\*e\_k(t) * s\_h(t)-\beta\*w\_{kh}(t)
\end{equation}
And using the maxtrix-vector notation,  

\begin{equation}
W(t+1) = W(t) + \alpha\*E_{o}(t) * s(t)^\mathrm{T}-\beta\*W(t)
\label{20}
\end{equation}

**$E_{o}(t)$** is a vector of $\{e\_k(t)\}$. It stands for the *errors* of the output. 


### Updating $U$ 

Similar to update $W$, we need to compute $\frac{\partial{L(t)}}{\partial{u\_{ji}(t)}}$ to change $U$. The chain of $u\_{ji}(t)$ is shown in Figure 3.
![Figure 3](/images/u_chain.png "Chain of U") 

The calculation is more complex than updating $W$. There are two sums here. But we don't need to calculate from the last step. 
Since we already know $\frac{\partial{L(t)}}{\partial{c\_{k}(t)}}$ in equation $\eqref{16}$, 
we can start from $\frac{\partial{c\_{k}(t)}}{\partial{s\_{j}(t)}}$ to speed our calculation.  
\begin{equation}
\frac{\partial{c\_{k}(t)}}{\partial{s\_{j}(t)}} = w\_{kj}(t)
\label{21}
\end{equation}

Then we can calculate $\frac{\partial{L(t)}}{\partial{s\_{j}(t)}}$, 
\begin{equation}
\frac{\partial{L(t)}}{\partial{s\_{j}(t)}} = \sum\_{k=1}^{V}{
	\frac{\partial{L(t)}}{\partial{c\_{k}(t)}} \*
	\frac{\partial{c\_{k}(t)}}{\partial{s\_{j}(t)}}
}
=\sum\_{k=1}^{V}{e\_{k}(t)\*w\_{kj}(t)} = E\_{o}(t)^\mathrm{T} \* W\_j
\label{22}
\end{equation}

$W_j$ denotes the $j$-th column vector of $W$.   
Continue going forward and by use of equation $\eqref{3}$ and equation $\eqref{22}$, we can get
\begin{equation}
\frac{\partial{L(t)}}{\partial{b\_{j}(t)}} = \frac{\partial{L(t)}}{\partial{s\_{j}(t)}} \* \frac{\partial{s\_{j}(t)}}{\partial{b\_{j}(t)}}
= E\_{o}(t)^\mathrm{T} \* W\_j \* s\_{j}(t) \* (1-s\_{j}(t))
\label{23}
\end{equation}

And now, we can compute $\frac{\partial{L(t)}}{\partial{u\_{ji}(t)}} $,
\begin{equation}
\frac{\partial{L(t)}}{\partial{u\_{ji}(t)}} =\frac{\partial{L(t)}}{\partial{b\_{j}(t)}} * \frac{\partial{b\_{j}(t)}}{\partial{u\_{ji}(t)}}
= E\_{o}(t)^\mathrm{T} \* W\_j \* s\_{j}(t) \* (1-s\_{j}(t)) \* x\_i(t)
\label{24}
\end{equation}

Therefore, we can update $u\_{ji}(t)$ by 
\begin{equation}
u\_{ji}(t+1) = u\_{ji}(t) + \alpha \* E\_{o}(t)^\mathrm{T} \* W\_j \* s\_{j}(t) \* (1-s\_{j}(t)) \* x\_i(t) - \beta \* u\_{ji}(t)
\label{25}
\end{equation}

However, this equation $\eqref{25}$ is too long. We want to make it like equation $\eqref{20}$ using the matrix-vector notation style.   

Compare to equation $\eqref{20}$, we need to ensure $E\_{o}(t)^\mathrm{T} \* W\_j \* s\_{j}(t) \* (1-s\_{j}(t))$ to be the $j$-th element of a certain vector. 
Suppose we call this vector as $E\_{h}(t)$ and $E\_{hj}(t)$ is the corresponding $j$-th element.   

And we define $d\_{hj}(E\_{o}(t)^\mathrm{T},t)$ to refer this calculation for convenience, which clearly means $E\_{hj}(t)=d\_{hj}(E\_{o}(t)^\mathrm{T}W,t)$.

\begin{equation}
d\_{hj}(E\_{o}(t)^\mathrm{T}W,t) = E\_{o}(t)^\mathrm{T} \* W\_{j} \* f^\mathrm{\prime}(s\_{j}(t)) 
= E\_{o}(t)^\mathrm{T} \* W\_j \* s\_{j}(t) \* (1-s\_{j}(t)) 
\label{26}
\end{equation}

And now the target matrix $E\_{h}(t)=d\_{h}(E\_{o}(t)^\mathrm{T}W,t)$.   
$E\_{h}(t)$ is a function of $E\_{o}(t)^\mathrm{T}W$. 
$E\_{o}(t)$ is the error of output layer. $E\_{o}(t)^\mathrm{T}W$ is the error propagated from output layer to hidden layer. 
Actually, $E\_{h}(t)$ is the error accumulated at the hidden layer. 
It seems error **propagated recursively** from output layer to hidden layer under the function $d_{h}$ and the matrix $w$ between these two layers. Interesting....!!!

Remember we want to obtain the maxtrix-vector form to update $U$. Using $E\_{h}(t)$, the updating rule is
\begin{equation}
U(t+1) = U(t) + \alpha \* E\_{h}(t) \* x(t)^\mathrm{T} - \beta \* U(t)
\label{27}
\end{equation}
### Updating $R$ 

Similarly to update $U$, we can get the updating rules of $R$,

\begin{equation}
R(t+1) = R(t) + \alpha \* E\_{h}(t) \* s(t-1)^\mathrm{T} - \beta \* R(t)
\label{28}
\end{equation}

---
##Backward Progagation Through Time

* Why BPTT?

* Unfolding  
![Figure 3](/images/unfold_rnn.png "Unfolding Recurrent Neural Network") 

* $\frac{\partial{L(t)}}{\partial{u\_{ji}(t-1)}}$

$x\_i(t-1)-->u\_{ji}(t-1)-->b\_{j}(t-1)-->s\_{j}(t-1)-->\sum{b\_k(t)}$

According to $\eqref{23}$, we all already know $\frac{\partial{L(t)}}{\partial{b\_{k}(t)}}$, 
\begin{equation}
\frac{\partial{L(t)}}{\partial{b\_{k}(t)}} = E\_{hk}(t)
\end{equation}

\begin{equation}
\frac{\partial{L(t)}}{\partial{s\_{j}(t-1)}} = \sum\_{k=1}^{H}{\frac{\partial{L(t)}}{\partial{b\_{k}(t)}} \* \frac{\partial{b\_{k}(t)}}{\partial{s\_{j}(t-1)}}}
= \sum\_{k=1}^{H}{E\_{hk}(t)*r\_{kj}(t)} = E\_h(t)^{\mathrm{T}} * R\_j
\end{equation}

~~~very similar to equation $\eqref{22}$, $R_j$ means the $j$-th column of matrix $R$. 

\begin{equation}
\frac{\partial{L(t)}}{\partial{b\_{j}(t-1)}}  = \frac{\partial{L(t)}}{\partial{s\_{j}(t-1)}} \* \frac{
\partial{s\_{j}(t-1)}
}{
\partial{b\_{j}(t-1)}
}
= E\_h(t)^{\mathrm{T}} * R\_j \* s\_{j}(t-1) \* (1-s\_{j}(t-1) ) 
\end{equation}

Using the function defined in $\eqref{26}$, we found
\begin{equation}
 E\_h(t)^{\mathrm{T}} * R\_j \* s\_{j}(t-1) \* (1-s\_{j}(t-1) ) = d\_{hj}(E\_h(t)^{\mathrm{T}}\*R, t-1)
\end{equation}

Therefore, 
\begin{equation}
 \frac{\partial{L(t)}}{\partial{b\_{j}(t-1)}} = d\_{hj}(E\_h(t)^{\mathrm{T}}\*R, t-1)
\end{equation}

For convenience, we define $E\_{hj}(t-1)=d\_{h,j}(E\_{h}(t)^\mathrm{T}R,t-1)$, and sequencely we know the vector $E\_{h}(t-1) = d\_{hj}(E\_{h}(t)^\mathrm{T}R,t-1)$.  

Then, 
\begin{equation}
 \frac{\partial{L(t)}}{\partial{u\_{ji}(t-1)}} = \frac{\partial{L(t)}}{\partial{b\_{j}(t-1)}} * \frac{\partial{b\_{j}(t-1)}}{\partial{u\_{ji}(t-1)}} 
 = d\_{hj}(E\_h(t-1)^{\mathrm{T}}\*R, t-1) * x\_{i}(t-1)
\end{equation}

Similar to equation $\eqref{27}$, we can get the target updating rule,

\begin{equation}
U(t+1) = U(t) + \alpha \* E\_{h}(t-1) \* x(t-1)^\mathrm{T} - \beta \* U(t)
\label{35}
\end{equation}

Simultaneously, 
\begin{equation}
R(t+1) = R(t) + \alpha \* E\_{h}(t-1) \* s(t-2)^\mathrm{T} - \beta \* R(t)
\label{36}
\end{equation}

Comparing these updating rules equation $\eqref{27}$ and equation $\eqref{35}$, equation $\eqref{28}$ and equation $\eqref{36}$,  we found they are in a **consistent** form. 
The calculation procedure has nothing special, except the value of some variables is different. 
Similarly, at time $t-2$, considering the cost function is $L(t)$, we can get these updating rules, 
\begin{equation}
 U(t+1) = U(t) + \alpha \* E\_{h}(t-2) \* x(t-2)^\mathrm{T} - \beta \* U(t) \\\
 R(t+1) = R(t) + \alpha \* E\_{h}(t-2) \* s(t-3)^\mathrm{T} - \beta \* R(t) 
\end{equation}

But what is $E\_{h}(t-2)$?   

Luckily, if we look the definition of $E\_{h}(t)$ carefully, we can find it is defined in a rescursive way.  
$E\_{h}(t)$ is a function of $E\_{o}(t)$.  
$E\_{h}(t-1)$ is a function of $E\_{h}(t)$.   
No doubtly, $E\_{h}(t-2)$ will be a function of $E\_{h}(t-1)$ and so on.  

Actually, $E\_{h-1}(t)$ means errors propagated from the output layer at time $t$ to the hidden layer at time $t-1$. Therefore we can say the error propagated in a recursive way.  
The matrix in function $d$ is the interaction matrix between the layer error comes in  and the layer flows out. 
For $E\_{h}(t)$, the interaction matrix  is $W$.   
For $E\_{h}(t-1)$, the matrix is $R$.   
For $E\_{h}(t-2)$ and later vector such as $E\_{h}(t-3)$, this martrix always is $R$. Because the error flows from current hidden layer to previous hidden layer and $R$ is the matrix between two hidden layers.

Therefore we can get the form of $E\_{h}(t-2)$,

\begin{equation}
	E\_{h}(t-2) = d\_{h}(E\_{h}(t-1)^{\mathrm{T}}R,t-2)
\end{equation}


* Weight Updating Mechanism   
There is still another question.  We already know derivative of $\frac{\partial{L(t)}}{\partial{u\_{ji}(t)}}$,
$\frac{\partial{L(t)}}{\partial{u\_{ji}(t-1)}}$, $\frac{\partial{L(t)}}{\partial{u\_{ji}(t-2)}}$. 
If we update them one by one, 
the value of $U$ differs with time $t$ in the unfolded network. 
But we must make sure $U$ will be the same at every time in the unfolded situation. 
how to update $U$?  

Suppose we have two variables, $M\_1$ and $M\_2$. 
 Their initial values are the same, say it is $M\_0$. At each step, we will update $M\_1$ and $M\_2$ by their derivatives $\frac{\partial{L}}{\partial{M\_1}}$
and $\frac{\partial{L}}{\partial{M\_2}}$ respectively.  
And we want to make sure the value of $M\_1$ and $M\_2$ are always the same in each step. But values of $\frac{\partial{L}}{\partial{M\_1}}$ and $\frac{\partial{L}}{\partial{M\_2}}$ usually are different.  How can we achieve this goals?  
A simple technique here is to put $\frac{\partial{L}}{\partial{M\_1}}$ and $\frac{\partial{L}}{\partial{M\_2}}$ together. And use the sum result of average result 
to modify $M\_1$ and $M\_2$ by the same amount. 
\begin{equation}
M\_1^{new} =M\_1^{old} + \alpha \* (\frac{\partial{L}}{\partial{M\_1}}+\frac{\partial{L}}{\partial{M\_2}}) \\\
M\_2^{new} =M\_2^{old} + \alpha \* (\frac{\partial{L}}{\partial{M\_1}}+\frac{\partial{L}}{\partial{M\_2}})
\end{equation}
Since their initial values are the same and their derivatives are the same, their values will always be the same. And this weight updating mechanism considers derivative contributions from both side. 

Now, we can get final updating rules of $W$, $U$, $R$, 
\begin{equation}
 W(t+1) = W(t) + \alpha \* E\_{o}(t) \* s(t)^\mathrm{T} - \beta \* W(t) \\\
 U(t+1) = U(t) + \alpha \* \sum\_{m=0}^{M}{E\_{h}(t-m) \* x(t-m)^\mathrm{T}} - \beta \* U(t) \\\
 R(t+1) = R(t) + \alpha \* \sum\_{m=0}^{M}{E\_{h}(t-m) \* s(t-m-1)^\mathrm{T}} - \beta \* R(t) 
\end{equation}

where $M$ is the steps for unfolding. 

##References
