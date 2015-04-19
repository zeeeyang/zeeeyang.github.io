---
layout: post
title: Recurrent Neural Network Language Model Implementation Step by Step
date:  2015-03-13 15:01 
categories: jekyll update
---
There are many papers, notes and blog posts for explaining theories and mathmatical details of recurrent neural network language model (rnnlm). 
However, there are few notes aimed at how to implement a rnnlm step by step. 
Implementing a recurrent neural network for a newbies is not very easy, especially when using a coding language like c/c++. 
There are many tricks we need to pay attention to. 
After understanding this tricks,  we can borrow them into our future works. 
I chose a piece of implementation code **rnnlm-0.4b** to understand every corner in it for serveral weeks and I learned many useful experience from this implementation.
Therefore, this is a blog for some coding details in **rnnlm-0.4b**.
##Entry Point: rnnlm.cpp/main
First, let's look at the main function of this toolkit.  
The main structure is shown as code below. 
"..." denotes currently not very important codes.  
{% highlight cpp %}
if(train_mode){
CRnnLM model1;
...
model1.trainNet();
}
if(test...){
CRnnLM model1;
...
if(nbest==0) model1.testNet();
else model1.testNbest();
}
if(gen...){
CRnnLM model1;
...
model1.testGen();
}
{% endhighlight %}
It contains four main functions:  
- **trainNet** training neural network parameters  
- **testNet** calculating perplexity of a test corpus given a rnnlm model  
- **testNBest** rescoring nbest sentences given a rnnlm model  
- **testGen** randomly generating plausible corpus given a rnnlm model    
I will cover both these functions.  
##Entry Point: rnnlmlib.cpp/trainNet
After having a basic understanding of the structure **rnnlm.cpp/main** function, we can find the training entry point.
Then we can jump to **rnnlmlib.cpp/trainNet**.
The main code structure of trainNet is shown as below.
{% highlight cpp %}
if(...){
   ...
}else{
  learnVocabFromTrainFile(); //build vocabulary
  initNet(); // init network structure and parameters
  iter = 0;
}
last_word = 0;
while( iter < maxIter ){
 ...
 while(1){
    ...
    word = readWordIndex(fi); //read next word
    computeNet(last_word, word); // forward compuation
    ...
    learnNet(last_word, word); // backward propagation
    ...
    last_word = word;
 }
 ...
 saveWeights(); //save parameters of network
 ...
 saveNet(); //save the whole neural network
}
{% endhighlight %}
We have five main procedures here:  
- **Build Vocabulary** read corpus word by word, turn each word into int and maintain structures to find words  
- **Initialize weights** randomly assgin values for network parameters  
- **Forward Computation** compute output through the network structure  
- **Backward Propagation** update parameters of neural network  
- **Save Model** save parameters of neural network to file  
##Building Vocabulary
###learnVocabFromFile
###addWordToVocab
###searchVocab
##Initilization: initNet
##Forward Computation: computeNet
##Backward Propagation: learnNet
##Review trainNet 
##testNet
##testGen
