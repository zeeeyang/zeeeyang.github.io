---
layout: post
comments: true
title: RNNLM Implementation (3) Initialize Network
date: 2015-04-20 10:30
categories: rnnlm
tags: 
- deep learning
---
In the last post, I described how to build vocabulary in rnnlm. 
In this post, we will know how to initialize the **structure** and **weights** of recurrent neural network.  
##Network Structure
Firstly, let us look at the network structure. 
There are two main structures in rnnlm. 
One is denoted as "simple" here, which is composed of three layers: input layer, hidden layer and output layer. 
The "simple" structure is shown in Figure 1. 
![Figure 1](/images/initNet/network_structure_simple.png "Simple Network Structure")
Another is denoted as "compression" here, which exploits a **compression layer** to speed up calculation.  
Suppose the hidden layer size is 100 and the output layer size is 10,000. 
Then we need to calculate $100 \* 10,000$ parameters. 
But if we use a compression layer with size 60. 
The number of parameters become $100\*60+60\*10,000$, which is smaller than $100 \* 10,000$.   
The "compression" structure is shown in Figure 2. 
![Figure 2](/images/initNet/network_structure_compression.png "Network Structure with Compression Layer")  
The code for initializing these two structures is shown below. 
{% highlight cpp linenos %}
layer0_size = vocab_size + layer1_size; 
layer2_size = vocab_size + class_size;
neu0 = ( struct neuron * ) calloc( layer0_size, sizeof( struct neuron ) );
neu1 = ( struct neuron * ) calloc( layer1_size, sizeof( struct neuron ) );
neu2 = ( struct neuron * ) calloc( layer2_size, sizeof( struct neuron ) ); 
neuc = ( struct neuron * ) calloc( layerc_size, sizeof( struct neuron ) );
syn0 = ( struct synapse * ) calloc( layer0_size * layer1_size , sizeof( struct synapse ) )
if (layerc_size == 0)
	syn1 = (struct synapse * ) calloc( layer1_size * layer2_size, sizeof ( struct synapse) )	
else{
	syn1 = (struct synapse * ) calloc( layer1_size * layerc_size, sizeof ( struct synapse) )
	sync = (struct synapse * ) calloc( layerc_size * layer2_size, sizeof ( struct synapse ) )
}
{% endhighlight %}
**struct neuron** is the structure to describe a neural node in the network.  
It has two attributes.  
- *ac* stores the value of forward computation.  
- *er* stores the value of backward propagation.  
**struct synapase** is the structure to store weights between two layers. 
One important attribute is:  
- *weight*. $syn_k[i][j].weight$ stores the value from node $i$ in layer $k+1$ to node $j$ in layer $k$.   

Only one variable *class_size* here needs further explaination. 
Why we need to make classes over word? 
One reason is to speed up the calculation on the output layer.   
Instead of directly calculating the probability $P( word | context )$, we predict word by $P(class | context ) * P ( word | class, context )$.  
##Make classes
But how to make classes over words?  
The author uses a **frequency-based** method. 
Suppose we are given $10$ words $w_1, w_2, \dots, w_{10}$ and the class size is $5$. 
The frequency of word counts is shown in table below.   
| $w_1$ | $w_2$ | $w_3$ | $w_4$ | $w_5$ | $w_6$ | $w_7$ | $w_8$ | $w_9$ | $w_{10}$ |
| --  | --  | --  | --  | --  | --  | --  | --  | --  | --   | 
| 40  | 30  | 10  |  5  | 5   |  2  | 2   | 2   | 2   | 2    | 
##Direct Connections
##BPTT Init
##Init Weight
