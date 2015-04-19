---
layout: post
title: RNNLM Implementation (2) Build Vocabulary
date:  2015-04-19 20:30
categories: jekyll update
---
In the previous post, I generally introduce the code framework of rnnlm. 
Especially I focus on the training part. 
Since in rnnlm the output layer is the whole vocabulary, we need to construct vocabulary firstly.  
In this post, I will introduce vocabulary building code in great details.   
This part of code is not the core of rnnlm, you can skip it for saving some time. 
But understanding this part will help us to know the semantic of some varaibles in the core part.   
###learnVocabFromFile
Similarly, let us look at the main code of function *learnVocabFromFile*.
{% highlight cpp linenos %}
addWordToVocab("</s>");
while (1){
	readWord(word, fin);
	if( feof(fin) ) break;
	i = searchVocab( word );
	if (i == -1){
		a = addWordToVocab(word);
		vocab[a].cn = 1;
	}else vocab[i].cn++;
}
sortVocab();
{% endhighlight %}
In Line 1, it adds a special mark "\</s\>" as a sentence boundary flag before reading file.   
In Line 3, it reads a *word* from file *fin*.  
In Line 5-9, it maintains an integer index of *word* as *a* and  collects the frequency of this word *vocab[a].cn*.   
In Line 11, it sorts the vocabulary by word frequency.  
###readWord(word,fin)
This function will read a word *word* character by character from file *fin*. 
Words could be separated by ' ' or '\t'. 
Each word is limited to a maximum length. Otherwise it will be truncated. 
At the end of a sentence, the sepcial mark "\</s\>" will be appended to this sentence.  
###searchVocab(word)
Function *searchVocab* will return the position of *word* in vector *vocab*. 
If not found, it will return -1.
{% highlight cpp %}
hash = getWordHash( word );
if ( vocab_hash[ hash ] == -1 ) return -1;
//equal
if ( !strcmp( word, vocab[ vocab_hash[ hash ].word ) ) return vocab_hash[ hash ];
//duplicate hash code, linear probing
for ( a = 0 ;  a < vocab_size; a++){
	if( ! strcmp( vocab[ a ].word, word) ) {
		vocab_hash[ hash ] = a;
		return a;
	}
}
return -1;
{% endhighlight %}
The idea is very simple here. It uses hash function to search a word. 
If there are collisions, it will use linear probing to find the real matched word. 
Below are meanings of some variables of functions used here.  
- **getWordHash** returns the hash code of a word.   
- **vocab_hash[ hash ]** stores the position of a word with hash code *hash* in vector *vocab*. Default value is -1.  
- **vocab[ a ].word** stores the word in position *a*.  
- **vocab_size** stores the number of distinct words in *vocab* currently.  
###addWordToVocab(word)
Add a word to vocabulary.
We need to maintain the structure of *vocab* and  *vocab_hash*.
{% highlight cpp %}
strcpy( vocab[ vocab_size ].word, word );
vocab[ vocab_size ].cn = 0;
vocab_size ++;
if( vocab_size +2 >= vocab_max_size ) {//reallocate memory
	vocab_max_size +=100;
	vocab = ( struct vocab_word *) realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
}
hash = getWordHash( word );
vocab_hash [ hash ] = vocab_size -1;
return vocab_size - 1; 
{% endhighlight %}
After knowing the meaning of structure *vocab* and *vocab_hash*, it is very easy to understand code here. 
###sortVocab
This function uses **selection sorting algorithm** to sort the vocabulary by word frequency *vocab[ word_index ].cn* from high to low.
Selection sort is a simple sorting algorithm and there is nothing specical to apply this algorithm to sort vocabulary. 
The main code is listed below for recap.
{% highlight cpp %}
for ( a = 1; a < vocab_size; a++) {
   max = a;
   for ( b = a+1; b < vocab_size; b++) {
	if ( vocab[ max ].cn < vocab[ b ].cn ) max = b;
   swap = vocab [ max ];
   vocab[ max ]  = vocab[ a ];
   vocab[ a ] = swap; 
}
{% endhighlight %}
I believe you are familiar with this algorithm, so no more explainations here. 
But why we need to sort the vocabulary? 
We will resolve this question in next post **InitNet**.
