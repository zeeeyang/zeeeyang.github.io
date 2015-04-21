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
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |   
| 40  | 30  | 10  |  5  | 5   |  2  | 2   | 2   | 2   | 2    |   

Remember that the vocabulary is already sorted by word frequency from high to low.   
The algorithm divides words into $5$ bins by code below. 
{% highlight cpp linenos %}
a = 0; df = 0;
for(i = 0; i < vocab_size; i++) b += vocab[i].cn;
for(i = 0; i < vocab_size; i++) {
    df += vocab[i].cn /(double) b;
    if( df > 1 ) df = 1;
    if( df > (a+1) /(double)class_size ){
        vocab[i].class_index = a; 
        if ( a < class_size -1  ) a++;
    } else {
        vocab[i].class_index = a;
    }
}
{% endhighlight %}
Firstly, it collects the total count of words, $b = 100$ here.   
Then, it calculate the accumulated frequency as $df$, shown as table below.  

| $w_1$  | $w_2$ | $w_3$ | $w_4$ | $w_5$ | $w_6$ | $w_7$ | $w_8$ | $w_9$ | $w_{10}$ |
| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | 
| $\frac{40}{100}$  | $\frac{70}{100}$ | $\frac{80}{100}$| $\frac{85}{100}$ | $\frac{90}{100}$|  $\frac{92}{100}$| $\frac{94}{100}$| $\frac{96}{100}$| $\frac{98}{100}$| $1$  |   

$df$ will be compared with $\frac{1}{5}$, $\frac{2}{5}$, $\frac{3}{5}$, $\frac{4}{5}$ and $1$.  

| word | $w_1$  | $w_2$ | $w_3$ | $w_4$ | $w_5$ | $w_6$ | $w_7$ | $w_8$ | $w_9$ | $w_{10}$ |
|:--------------| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | 
| df | $\frac{40}{100}$  | $\frac{70}{100}$ | $\frac{80}{100}$| $\frac{85}{100}$ | $\frac{90}{100}$|  $\frac{92}{100}$| $\frac{94}{100}$| $\frac{96}{100}$| $\frac{98}{100}$| $1$  |   
|Compared to| $\frac{1}{5}$ | $\frac{2}{5}$ |$\frac{3}{5}$ |$\frac{4}{5}$ |$1$ |$1$ |$1$ |$1$ |$1$ |$1$ |
|class_index| 0 | 1 | 2 | 3| 4| 4| 4| 4| 4| 4| 

This algorithm tends to make high frequency word into a small class and low frequency into a large class. For example, $class_0$ only has one element and $class_4$ has six elements.   
There is an another algorithm for the same purpose. The previous approach is called as "old_classes". In "new_classes" algorithm, the $df$ claculation is different. 
{% highlight cpp linenos %}
a = 0; df=0; dd = 0;
for(i = 0; i < vocab_size; i++) b += vocab[i].cn;
for(i = 0; i < vocab_size; i++) dd += sqrt(vocab[i].cn/(double) b);
for(i = 0; i < vocab_size; i++) {
    df += sqrt(vocab[i].cn /(double) b ) / dd ;
    if( df > 1 ) df = 1;
    if( df > (a+1) /(double)class_size ){
        vocab[i].class_index = a; 
        if ( a < class_size -1  ) a++;
    } else {
        vocab[i].class_index = a;
    }
}
{%endhighlight%}
In Line 5, we use $\frac{\sqrt{ \frac{vocab[i].cn}{b}}}{dd}$ to replace $\frac{vocab[i].cn}{b}$.   

| word | $w_1$  | $w_2$ | $w_3$ | $w_4$ | $w_5$ | $w_6$ | $w_7$ | $w_8$ | $w_9$ | $w_{10}$ |
|:--------------| :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | 
| df | 0.2386 | 0.4452 | 0.5645 | 0.6488 | 0.7332 | 0.7865 | 0.8399 | 0.8932 | 0.9466 | 1.0 |
|Compared to| $\frac{1}{5}$ | $\frac{2}{5}$ |$\frac{3}{5}$ |$\frac{4}{5}$ |$1$ |$1$ |$1$ |$1$ |$1$ |$1$ |
|class_index| 0 | 1 | 2 | 3| 3 | 3 | 4| 4| 4| 4| 

We can see something different now. 
The class index of word $w_5$ and $w_6$ changes to 3. 
This algorithm can make word distributed less skewed. 
It makes some tradeoff between frequency distribution and number of words in a class. 
The "new_classes" algorithm is faster than the "old_classes" algorithm for later network calculations. 
##Init Weight
###neuron 
For each neuron, neuron.ac = 0 and neuron.er = 0. 
###synapse
For each synapse, syn.weight = random(-0.1,0.1) + random(-0.1, 0.1) + random(-0.1, 0.1)
plans
##Direct Connections
##BPTT Init
