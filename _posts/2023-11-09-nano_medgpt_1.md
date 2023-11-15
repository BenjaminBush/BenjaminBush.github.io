---
layout: post
title: Nano Med GPT Part 1
date: 2023-11-09 15:09:00
description: Part 1 of the Nano Med GPT series
tags: llm gpt medical_notes
featured: true
---
# Nano Med GPT Series
[Nano Med GPT](https://github.com/BenjaminBush/nano_medgpt/tree/main) is a small-scale version of the GPT-2 language model trained on medical note data. Although there are more advanced tools available, such as [DAX Dragon Express](https://www.nuance.com/healthcare/ambient-clinical-intelligence/explore-dax-for-clinicians.html) (which help provide inspiration for this project), this important use case presents a strong opportunity for hands-on learning and development with transformer-based DNN architectures. This blog series will be broken into the following parts:
* Part 1 - Problem Statement, Dataset, and Local Machine Development
* Part 2 - Scaling Up DNN on Azure, Effectiveness

# Problem Statement
Clinicians spend nearly 2 hours per day outside of regular working hours writing notes in a patient's chart. This time consuming task can lead to clinician burnout. Generative AI can assist with documentation by suggesting autocompletion, which may reduce the time required to complete a note and therefore alleviate some burden on the clinician. 

# Dataset
For this project, we make use of the [MIMIC-IV Dataset](https://physionet.org/content/mimiciv/2.2/). MIMIC-IV is the latest version of a database comprising the deidentified health-related data from patients who were admitted to the critical care units of the Beth Israel Deaconess Medical Center between 2008-2019. I am grateful to the producers and owners of the database for providing access. Those that are interested in gaining access to the dataset can follow the instructions detailed [here](https://mimic.mit.edu/docs/gettingstarted/). 

MIMIC-IV provides a wealth of interesting data for myriad research purposes. For this project, we are interested in the ```mimiciv_note``` module, which includes deidentified free text clinical notes for hospital details. We are specifically intersted in the discharge summaries, which can be found in the ```discharge``` table. 
## Exploratory Data Analysis
number of notes
distribution of note length
total raw vocabulary size (words, characters)
what data is de-identified
what is the general format of the note?

# Local Development
To develop the inital model, I followed [Karpathy's YouTube tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=5974s) in which he builds a GPT from scratch using the Shakespeare data. In my case, of course I used the ```mimiciv_note``` data. Karpathy provides an excellent overview of language modeling and transformer architecture and I highly recommend readers to visit his channel, or his blog. 

Given we are working on a "nano" scale of language modeling, we formulate our learning task as next character prediction. Observe that this differs from some of the current standard of predicting the next word, or sub-word. We elect this choice primarily due to compute and data constraints; we lack sufficient compute to be able to handle next word prediction and similarly our dataset is not large enough to meaningfully learn the relationships between all of the words in the corpus. 

## Data Preprocessing
As we discoverd in our exploratory data anlysis, the medical note data is messy! Before we begin any model development, it will be important to clean the dataset and normalize the inputs to ensure as smooth of a training process as possible. 

### Extracting from Google BigQuery
The ```mimiciv_note``` dataset is located in BigQuery. Fortunately, BigQuery allows you to export the results of a query to Google Cloud Storage. A simple:<p>
``` SELCT text FROM discharge;``` <p>
run against the ```mimiciv_note``` database will produce the desired result. We can then access the Google Cloud Storage bucket in the browser and download the uncompressed data. Given the size of the data, Google split the download into 7 files. For ease of reference, we refer to each of these files as "parts"; "Part 0" refers to the first chunk of the results returned from the query, "Part 1" contains the next chunk, and so on. They are approximately equal size in their row count. We will ultimately use Parts 0-5 for training and Part 6 as our test set.  This is approximately an 80/20 split. However, for local development we can only utilize the first // HOW MANY ROWS? of Part 0 due to local machine memory constraints. 

### Text Cleaning
Data and compute restrictions influenced our choice to formulate our task as next character prediction instead of next word prediction. These constraints will also be influential in our text cleaning methodology. 

The first preprocessing step is to lowercase all of the text. We lack sufficient data to learn representations between proper nouns, new sentences, etc. Furthermore, upon inspection of some of the notes, there are erroneous capitalizations of words. Lowercasing all of the words helps us begin the normalize the text for the machine to more easily understand. 

We also discover that the the text from the medical notes either includes some special characters, or was included into the dataset with different encodings. It will be important to make sure we use the same text encoding throughout handling the data. To that end, we first encode the raw bytes into ascii, ignoring any characters that do not fit into the ascii character set, and then decode into utf-8. 

Finally, we want to restrict our vocabularly to exclude special punctuation and other characters. We only desire to have alphanumeric characters and spaces to minimize the complexity of the text. To that end, we can write two simple regular expressions to remove unwanted text. 

When we put this all together, our function for cleaning text looks like this:
```python
def clean_text(text):
    text = text.lower()
    text = text.encode('ascii', 'ignore').decode("utf-8")
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.replace("\n", " ")
    return text
```

Unfortunately, we cannot load the entire dataset into memory and clean it in one shot. In fact, we can hardly load one Part into memory and perform basic data manipulation on it! To that end, we will need to write a small wrapper function to chunk the raw data, clean it one batch at a time, and write each cleaned batch to memory for future use. 

```python
def clean_text_chunked(text, chunk_size=10000):
    cleaned_text = ""
    start = 0
    n_chunks = len(text)/chunk_size + 1
    while start < len(text):
        chunk = text[start:start+chunk_size]
        cleaned_chunk = clean_text(chunk)
        cleaned_text += cleaned_chunk
        start += chunk_size
    return cleaned_text
```
Our data is now processed and almost ready to be used in training. 

## Encoding Text to Numbers
Machines can't understand words the way that humans do. We need to be able to represent all of the clean medical note text into something that a machine can understand. This task of creating machine-readable representations of text is often referred to as "encoding", and the machine-readable representations are referred to as "encodings" or "embeddings". 

Encodings have a very rich history and they're worth talking about a little more in depth because of their profound impact on the learning task. 

If you were to devise a scheme to create embeddings, a natural thought would be to have words that have similar meaning are "closer" in vector space (e.g., cosine similarity ~ 1). The classic example of this would is that "King" and "Queen" are conceptually similar and you could imagine that adding the vector word embedding of "woman" to the vector word embedding of "King" would produce a vector that is very similar to the vector word embedding of "Queen". In other words, 
$$\vec{woman} + \vec{King} = \vec{Queen}$$
This is concept is applied in the "Word2Vec" embedding scheme, which was first introduce in 2013. However, this approach suffered significant drawbacks. 
- Defining the vectors for each word is cumbersome. 
- Handling minor variation in words is difficult (e.g., through verb conjuction, or words that share similar stems)
- The notion of similarity in the vector space falls apart for most of the words in the dictionary. 
While one may conclude that $$\vec{fruit} + \vec{yellow} = \vec{banana}$$, what does $$\vec{fruit} + \vec{red} = ?$$ Strawberries? Raspberries? Something else entirely?<br>

This ultimately led to two key developments in NLP. The first is to let the machine create better embeddings as part of the learning process. The second was an innovation in the way we normally do tokenization; by splitting words into "sub-words", we can pass on some of the linguistic properties of words to the machine. For instance, the word "running" might be broken into subwords, ["run", "ning"]. The machine can easily understand "run" as second subword indicates the conjugation of the verb. The same concept applies to non-verbs, e.g., "fireplace" becomes ["fire", "place"]. 


There are myriad open source models that will produce embeddings for you. OpenAI also offers an Embedding API for you to leverage the same embeddings used in ChatGPT. However, we'll opt for something simpler. After all, part of the motivation for this project is to get our hands dirty!


Recall that our basic task in creating embeddings is to produce a mapping of characters to numbers (integers). 

To do this, we can simply iterate over our vocabulary and assign an integer to the index of a given character.


```python
char_vocab = list("0123456789abcdefghijklmnopqrstuvwxyz ")
stoi = {}
itos = {}
i = 0
for c in char_vocab:
    stoi[c] = i
    itos[i] = c
    i+=1
```

From here, we can create simple "encode" and "decode" functions by using the Python lambda operator. 


```python
encode = lambda s: [stoi[c] for c in s]
decode = lambda i: ''.join([itos[j] for j in i])
```

As an example, the sentence "Huck is sleeping next to the fireplace" can be encoded as<p> [17, 30, 12, 20, 36, 18, 28, 36, 28, 21, 14, 14, 25, 18, 23, 16, 36, 23, 14, 33, 29, 36, 29, 24, 36, 29, 17, 14, 36, 15, 18, 27, 14, 25, 21, 10, 12, 14]<br>
 Observe that the capital "H" in "Huck" first needs to be lowercased to "h" since our vocabulary is produced from our clean_text function. <br><br>
 Now that we have successfully cleaned our dataset, and creating a way to represent that dataset to the machine, we can begin developing a machine learning model to predict the next character!

## Language Modeling
Let us discuss in greater detail what "predict the next character" means. For simplicity, I have excluded the words "given some context". In this case, "some context" just means the previous characters. It's easy enough to think about this like an autocomplete task -- when you are searching in your search bar, or writing an e-mail, your browser will often suggest an autocomplete to help you achieve your goal more quickly. That is exactly what we aim to do here. Thus, we revise the formal statement of our task to<br>
Predict the next character given the previous $k$ characters. <br>
Where $k \geq 0$. <br>

For those of you that have used ChatGPT, you supply context in the form of the chat prompt. 

Let's aim to formulate this mathematically. When we seek to predict the next character, there is some probability associated with every character in our vocabulary. 
We formulate this probability of character $x$ being the next character given context $k$ as <br>
$$ P(x \mid k)$$
Using Bayes' Theorem, we can rewrite this as
$$ P(x \mid k) = \dfrac{P(k|x)P(x)}{P(k)} $$ 
Let's briefly examine each of the terms.
- $ P(k \mid x) $ is the likelihood. It is how likely the context is, given the next character. 
- $ P(x) $ is the prior probability of the next character. This can be directly calculated from the dataset (how often does charcter $x$ occur?)
- $ P(k) $ is the prior probability of the context. This can be directly calculated from the dataset. 

Something important to note here is that so far we have only specified the constraint $k \geq 0$. Although theoretically $k$ does not need to have an upper bound, in practice it does -- we do not have unlimited memory and if we allow $k$ to get big enough, then even if we could train a model that big the resulting model would likely just be reguritating information it had already seen. For local development, we bound $ 0 \leq k \leq 8$ and revist this choice in the second part of the blog.

Observe that the NLP task essentially boils down to estimating $P(x \mid k)$, or $P(k \mid x)$, which is a probability distribution. The model we develop will seek to learn this probability distribution and then we can sample from this distribution to generate new text. That is why this class of models are considered "Generative AI". 
* Not only can they "generate" new text, but 
* Compared to traditional discriminative models that estimate $P(x \mid k)$ directly, generative models traditionally estimate $P(k \mid x)P(x)$. 

Let's take our theoretical understanding of the language modeling task and apply it to a simple bigram model first. 

### Bigram Model
Recall that a bigram model is a specific type of n-gram, where n=2. It is also known as a bag of words model (or in our case, bag of characters!). In this case, it fixes the context, $k = 2$; in other words, given the previous two characters, predict the next character. 

We can visualize this as a 2D array where of size (vocabulary_size x vocabulary_size). The entries represent the counts of how frequently any pair of characters occurs, and then normalizing these counts into a probability distribution. This is easily achieved through the PyTorch ``nn.Embedding`` layer. 

We need to implement two functions for this model to be trained -- a "forward" method for the forward pass of training, and a "generate" method for generating text. 

The forward method is quite simple. We simply perform a lookup from our Embedding table (remember these are the counts of occurrences) and then compare the entry of the lookup table to the true value (what the next character actually is). We can use the cross-entropy loss funciton to calculate loss and let PyTorch handle the backpropagation step. 

The generate method follows what we have described above. Given some context (two characters), we'll first perform a lookup to get the model's prediction for the next character based on counts. From there, we'll normalize those counts into a probability distribution and then sample using the PyTorch's multinomial function. We then take the predicted character and append that to our context, using that for the next step in the generation sequence. We will iterate this process for a fixed number of tokens. 

Ultimately, the BigramLM class looks something like 

```python
class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
    
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # forward
            logits, loss = self(idx)

            # last timestep
            logits = logits[:, -1, :]

            # get all probabilities
            probs = F.softmax(logits, dim=-1)

            # sample from the probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled idx to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
```
You may notice there are some terms included in the forward method that we did not explicitly talk about, such as "B", "T", and "C". These are dimensions of the tensors and batch size used to help speed up training. Karpathy talks about them in depth during the YouTube tutorial and I would recommend checking that out for an in-depth view. 

We can go ahead and generate samples from the untrained Bigram LM to see what it will spit out prior to any training. 

> 0l6fu1be10mrb3e97hmdbmfdn7afclyax 87 3b2i1 b17az3 8qyacp30u4ty3xfe0zlhpn8817wqq5xsilhpo3bnirffzb8o2mx

>

As expected, it's basically a uniform prediction of a random characters. This would hardly be helpful to the already tired clinician. Let's see if the Bigram LM can do any better after training for 500 epochs.

> 0p d1hofcofs3rdaexrktapk9 tjca7rv5awzu6xoklclrdroknoghevy6iyzdrvcv5901udzerklhko7 3c683w7mcip t057cof
>

It's clear both the validation loss (3.49) and the samples produced that the Bigram LM lacks sufficient representational capacity to meaningfully learn the underlying relationships between the text in the corpus. It is not well suited to this task and we require a heavier hammer. 

## Transformers
GPTs are "Generative Pre-trained Transformers"; I would be remiss if I did not spend some time talking about transformers as they are the critical building block in this network architecture. More importantly, they are one of the innovations that has led to drastic improvements in NLP over the past 6 years and the same concepts are being applied for massive gains in computer vision currently. Transformers will be the key tool that allow us to produce a coherent model. 

A high level overview of our nano GPT model architecture is presented below. 
 ![GPT Architecture](/assets/img/nano_medgpt/gpt.png) <br>
 I assume the reader is familiar with Linear layers and LayerNorm layers, as these are standard in most modern Deep Learning toolkits, such as PyTorch. Let's break down each of the remaining components of the network starting with the Positional and Token Embedding inputs. 

### Embedding Inputs
The Token Embedding input is exactly what we described during the construction of the Bigram LM. It is simply a lookup table (represented in PyTorch's ```nn.Embedding``` module). The Positional Embedding Layer is also a ```nn.Embedding``` module. This embedding encodes "positional" information -- i.e., what index (position) a token is for a given sequence. For instance, if our context length $k = 7$, then the sequence "hi huck" will encode positions [h, 0], [i, 1], [ , 3], [h, 4], [u, 5], [c, 6], [k, 7]. Of course we will represent the tokens (characters) as numbers as we discussed before. Encoding both positional information and token information gives us everything we need to feed the network. 
### Feed Forward
The feed forward network is a Sequential block of multiple simple layers, as shown in the figure below ![Feed Forward Architecture](/assets/img/nano_medgpt/feed_forward.png) <br>
A token is fed into a Linear layer, followed by ReLU activation, and then into another Linear layer. The final dropout layer is included as a form of regularization to help prevent overfitting. Dropout layers are included in the rest of the diagrams, but are not strictly required, nor used in the original network design. In practice, however, they are quite beneficial to help make the network more robust. 
### Transformer Block
The transformer block is shown in the figure below.
![Transformer Block Architecture](/assets/img/nano_medgpt/transformer_block.png)<br> 
A token, x, is first normalized through LayerNorm, then passed into a series of MultiHead Attention blocks. We will discuss attention in the following sections. For now, you can think of the MultiHead Attention blocks as the key component to help the network learn the relationships between various tokens - over time it will help encode general linguistic properties of the English language and domain-specific relationships. After this key information is extracted from the MultiHead Attention blocks, we again normalize with LayerNorm and pass to a FeedForward network (described above). 
### MultiHead Attention
The MultiHead Attention block is shown in the figure below. ![MultiHead Attention Block Architecture](/assets/img/nano_medgpt/mha.png)<br>
The name "MultiHead" attention naturally begs the question, "What is Single Head Attention?". You guessed it -- that's our next (and final) deep-dive in this network architecture overview. We described MultiHead Attention as the key component to help learn relationships between various tokens; for now, you can think of a single Self Attention Head as the atomic component that learns different parts of these relationships (i.e., the likelihood from the above Bayes equation!). These different parts (outputs of the individual attention heads) are concatenated and passed to a Linear Layer. We again include a Dropout layer for regularization. 
### Self Attention
Self Attention is really the meat of the conversation when talking about Transformers. You can see from the diagrams above that every layer we've gone a little deeper has led us to this crucial block. Let's start with an overview of the Self Attention block shown in the figure below. ![Self Attention Block Architecture](/assets/img/nano_medgpt/sa.png)<br> 

Observe that there are three inputs: Q, K, and V. These stand for "Query", "Key", and "Value". Recall that the network receives Position and Token Embeddings from every single token. <br>

Every single token (character), in every single position, will emit two vectors: query and key. 
- The query vector can be interpreted as what that token is looking for. 
- The key vector can be interpreted as what that token contains. <br>

For instance, the token "s" may be interested in understanding what types of characters came before it, as the presence of the "s" at the end of a word could represent a plural (e.g., "shoe**s**"). We take the dot product (matrix multplication) of these two vectors, and then scale the result. The dot product between every key and every query represents the affinities across each of the tokens in a sequence. If the dot product is high between two tokens, they they have high affinity and will interact closely. Observe how this is similar to the "learning the relationships between each tokens". 

We include a masking operation to make sure that tokens at the $i$ th position can only look at the tokens at positions before $i$. We are focused on next character (token) prediction for this task, so allowing each token to look into the future before making its prediction may be cheating. <br>
We then pass the output through a SoftMax layer to obtain the weights on the values. We again include a Dropout layer for regularization. 

In practice, the key and the query vectors are simple Linear layers in the network. 

The final piece to the self-attention layer is a vector commonly referred to as "value", which is also a Linear layer. The value vector can be interpreted as the information that a given token will communicate to other tokens.  

We pass the output from the Dropout layer into one final matrix multiplication between the weights and the value vector. <br>
The authors of the original Self Attention paper, titled "Attention is All You Need", summarize this quite nicely -- 
> An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatability function of query with the corresponding key. 
>

Let's try and tie this back together with our Bayes' equation from above. We said we were really interested in estimating the likelihood $ P(k \mid x) $ and the prior $P(x)$. Recall that the likelihood is how likely the given context is given the next token, and the prior $P(x)$ is the prior probability of the next character. From our token embedding tables, we've gotten pretty good at estimating $P(x)$. Perhaps most interesting, however, is that we've created numerous interactions in our network between the context and the next token through attention (and masking) and then stacked many dense Linear layers on top of these interactions. In other words, the Self Attention Heads, coupled with Linear layers, help us better estimate the likelihood $P(k \mid x)$. Neural networks, particularly deep networks, are black boxes and very difficult to explain. However, I find that the visual representation of the network coupled with developing some mathematical intuition behind the individual layers helps make clear what the network is doing during each step of training. 

### Training the Nano GPT Model
I have very intentionally left unspecified the size of the network up to this point. We have ommitted the number of Transformer Blocks, number of MultiHead Attention Blocks, and number of Self-Attention Heads. Moreover, we have not discussed the dimensions of the hidden layers of the network, the learning rate, the batch size, or the context length. Recall that we are still developing on a local machine with very real physical constraints and your local system's constraints will likely dictate choice in hyperparamters for local development. Please refer to the GitHub repository for the latest on the hyperparamters used for my local development. We will discuss these parameters in more detail (and scaling them up!) in Part 2 of the blog series when we move to the cloud, but for now, let's just see how the model performs after a few epochs of training. The validation loss reaches around 1.95 after 500 epochs of training a fairly small GPT model.

```python
print(generate_text(model, 100))
```
> 0 mg od  ne1 8 tablets t by toleut 1406 1 on oftopy  4na discre 8 reur neg theout getly  earvers ri c
>

What about when we supply the first few words for the model and have it attempt to complete the rest?
```python 
print(prompt(model, "Patient History: Ben is "))
```
>patient history ben is sight of his ange heealve oter  hp wecteransa pormassions   reaniom   ht  his ov sup in af repentgio
>

Believe it or not, we're actually starting to get words that belong in the English dictionary! Most of the output is still garbled and nonsensical, but we have developed the critical building blocks for our language model. Seeing real words produced by a model trained only for ten minutes on a tiny amount of data while trying to predict the next character (not even a whole word!) is quite exciting. 

# Wrapping Up
Thanks for joining! Please stay tuned for Part 2 of the blog series where we will cover how to scale our model size and dataset size up using the cloud, as well as evaluate the effectiveness of our model. 