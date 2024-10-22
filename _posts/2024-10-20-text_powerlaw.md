---

title: "Power Law Distribution: Word Frequency"
date: 2024-10-20
typora-root-url: ./..
---



In a [previous note](https://kezhaozhang.github.io/2024/10/07/powerlaws.html), we explored one mechanism that leads to power law distributions: the probability of a random walk returning to its starting point for the first time. In this note, we will examine another mechanism that generates power law distribution of the word frequencies within a text.



### Word Frequency Distribution

The distribution of word frequency-essentially, how often words appear in a text-can be effectively modeled by a power law distribution, 

For instance, consider the text of the US Constitution. Figure 1 illustrates a word cloud, where the size of each word corresponds to its frequency of occurrence in the text. 



<figure>
  <center>
  <img src="/assets/images/zipf_wordcloud.svg" width="400">
   </center>
  <center>
    <figcaption> Figure 1. Word cloud of the US Constitution, with word size proportional to frequency.
    </figcaption>
  </center>
</figure>




Figure 2 presents the word frequency distribution, which aligns closely with the Zipf distribution well.

<figure>
  <center>
  <img src="/assets/images/zipf_histogram_fit.svg" width="500">
   </center>
  <center>
    <figcaption> Figure 2. Word frequency distribution compared with the Zipf distribution.
    </figcaption>
  </center>
</figure>




The probability density function (PDF) of the fitted Zipf distribution is given by:



$$
0.54 x^{-1.82},\notag
$$



where integer $x\ge 1$ denotes the frequency of word appearances.  Figure 3 demonstrates the strong correlation between the cumulative distribution function (CDF) of the observed data and the CDF of the fitted Zipf distribution.



<figure>
  <center>
  <img src="/assets/images/zipf_probability_plot.svg" width="400">
   </center>
  <center>
    <figcaption> Figure 3. CDF of the observed data versus CDF of the fitted Zipf distribution.
    </figcaption>
  </center>
</figure>




The general form of the PDF of the Zipf distribution is:



$$
\frac{x^{-\rho -1}}{\zeta(\rho +1)}, \notag
$$



where $x\ge 1$ and $\zeta$ is the Riemann zeta function defined as $\zeta(s)=\sum_{n=1}^{\infty}\frac{1}{n^s}$.



Figure 4 illustrates the relationship between word frequency and rank,  highlighting the first few words from the US Constitution in red. 



<figure>
  <center>
  <img src="/assets/images/zipf_counts_vs_rank_callout.svg" width="500">
   </center>
  <center>
    <figcaption> Figure 4. Word frequency versus rank,  with the first few words from the US Constitution highlighted.
    </figcaption>
  </center>
</figure>




The straight line in Figure 4, plotted on a logarithmic scale for both axes, indicates a power function relationship.  Figure 5 further illustrates that a power function fits the word frequency versus rank data, with the exponent approximately equal to $1$.  This aligns with Zipf's empirical law, which posits that a word's frequency is inversely proportional to its rank.  



<figure>
  <center>
  <img src="/assets/images/zipf_counts_vs_rank_with_fit2.svg" width="500">
   </center>
  <center>
    <figcaption> Figure 5. Word frequency versus rank with a power function fit.
    </figcaption>
  </center>
</figure>




The table below compares the word frequency distribution of the three texts. In each case, the distribution is best modeled by the Zipf distribution. Notably, the exponent $\rho$ of the Zipf distribution increases as the ratio of unique words to total words rises. 

|        Text         | Ratio of Unique Words to Total Words | Exponent $\rho$ of Zipf fit |
| :-----------------: | :----------------------------------: | :-------------------------: |
|   US Constitution   |                 0.16                 |            0.82             |
|  Origin of Species  |                0.048                 |            0.58             |
| Pride and Prejudice |                0.054                 |            0.61             |




### Simon Model

Herbert Simon proposed a model to explain the emergence of power-law distribution in text frequency:



Words are generated one at a time. Each added word can either be a new word (with probability $\alpha$) or one of the existing words (with probability $1-\alpha$). 



Assuming there are $N$ total words and $n$ unique words in the text, the process to add the next word proceeds as follows: with probability $\alpha$, a new word is introduced, resulting in the text with $N+1$ words and $n+1$ unique words. Alternatively,  with probability $1-\alpha$, an existing word is selected with probability proportional to its frequency, keeping the total words at $N+1$ words and $n$ unique words unchanged.



We implement this algorithm using  Wolfram Language to simulate the word frequency distribution. Initially, $100$ words were generated to kickstart the text generating process. Each word is represented by an  integer drawn randomly from a discrete uniform distributed between $1$ and $100$.



The initial words are stored in an association array, with keys are word IDs and values indicating their frequency of appearance. The following code snippet creates this initial array: 



```mathematica
SeedRandom[1234];
init =Association@Map[#[[1]]->#[[2]]&, 
                       Tally[RandomVariate[DiscreteUniformDistribution[{1, 100}], 100]]
                       ];
```



Figure 6 showcases an example of these initial words along with their frequencies. 

<figure>
  <center>
  <img src="/assets/images/zipf_simulation_initial_condition.svg" width="400">
   </center>
  <center>
    <figcaption> Figure 6. Initial words and their frequencies.
    </figcaption>
  </center>
</figure>




The following code executes the Simon model to simulate the generation of $10000$ words with $\alpha=0.05$.


```mathematica
NextWord[counter_, alpha_]:= Module[{dict = counter, newword, key},
    If[RandomReal[]<alpha,
        (* with probability alpha, add a new word *)
        newword = Max[Keys[dict]]+1;
        dict = Append[dict, newword->1], 
        (* otherwise, randomly pick an existing word with probability proportional to its frequency*)
        k = RandomChoice[Values[dict]->Keys[dict]];
        dict[[Key[k]]] = dict[[Key[k]]]+1]; (*selected word count increased by 1)
    dict
    ]
```



```mathematica
alpha=0.05;
steps = 10000; (* number of steps in the text generation *)
distribution = Nest[NextWord[#, alpha]&, init, steps]; (* final word distribution *)
```



Figure 7 displays the word frequency distribution for varying values of $\alpha$. The straight line in the log-log histogram indicates a power law distribution when $\alpha>0.025$.

<figure>
  <center>
  <img src="/assets/images/zipf_simulation_distribution_vs_alpha.svg" width="700">
   </center>
  <center>
    <figcaption> Figure 7. Word frequency distribution for different values of <i>&alpha;</i>.
    </figcaption>
  </center>
</figure>




The table below shows the fit of the word frequency distribution for various $\alpha$ values, as determined by the `FindDistribution` function in Wolfram Language. For $\alpha=0.025$, the best fit is a mixture of Zipf and Geometric distributions.  For $\alpha>0.025$, the Zipf distribution provides the best fit. The parameter $\rho$ of the Zipf distribution is calculated for each $\alpha$.  The monotonic relationship between $\alpha$ and $\rho$ is consistent with patterns observed in actual texts like the US Constitution, "Origin of Species", and "Pride and Prejudice". 

| $\alpha$ | Fit to Zipf Distribution $\rho$ |            Best fit Distribution            |
| :------: | :-----------------------------: | :-----------------------------------------: |
|  0.025   |              0.52               | Mixture of Zipf and Geometric distributions |
|   0.05   |              0.65               |                    Zipf                     |
|   0.10   |              0.76               |                    Zipf                     |
|   0.15   |              0.85               |                    Zipf                     |
|   0.20   |              0.89               |                    Zipf                     |
|   0.25   |              0.94               |                    Zipf                     |





<figure>
  <center>
  <img src="/assets/images/zipf_simulation_histogram_vs_alpha_with_zipf_fit.svg" width="700">
   </center>
  <center>
    <figcaption> Figure 8. Word frequency distribution with fits to the Zipf distribution.
    </figcaption>
  </center>
</figure>




The goodness of fit of the Zipf distribution to the word frequency data is illustrated in Figure 9, where the correlation of the CDF of word frequency and the CDF of the Zipf distribution is plotted. When $\alpha=0.025$, noticeable deviations from the one-to-one line appear in the tail region, indicating that the acutal data does adhere strictly to the Zipf distribution. A similar, albeit weaker, deviation is evident for $\alpha=0.05$, while the correlation is nearly perfect for $\alpha \ge 0.1$.

<figure>
  <center>
  <img src="/assets/images/zipf_simulation_probabilityplot_vs_alpha.svg" width="700">
   </center>
  <center>
    <figcaption> Figure 9. CDF correlation between word frequency and Zipf distribution for various <i>&alpha;</i>.
    </figcaption>
  </center>
</figure>




 Figure 10 compares the tails of the word frequency distributions for various $\alpha$ values. A smaller $\alpha$ leads to a smaller $\rho$ for the fitted Zipf distribution, resulting in a longer tail. 

<figure>
  <center>
  <img src="/assets/images/zipf_simulation_cdf.svg" width="500">
   </center>
  <center>
    <figcaption> Figure 10. CDFs of word frequency distributions for different <i>&alpha;</i> values.
    </figcaption>
  </center>
</figure>




This phenomenon occurs becaue a smaller $\alpha$ results in a higher likelihood of generating existing words during text generation. Consequently, more words accumulate higher frequencies, contributing to a longer tail in the distribution.



Regarding the relationship between word frequency and rank, Figure 11 demonstrates that for rank greater than 10, this relationship typically follows a power law, as indicated by the straight lines in the log-log plot. However, the simulated data deviates from the power law for lower rank values, with deviations being more pronounced when more initial words are used. 



<figure>
  <center>
  <img src="/assets/images/zipf_simulation_freq_vs_rank_alpha0.15.svg" width="500">
   </center>
  <center>
    <figcaption> Figure 11. Word frequency versus rank for different conditions (10 and 100 initial words), with <i>&alpha;=0.15</i>. The straight lines represent the best power function fits: <i>y=841x<sup>-0.96</sup></i> and <i>y=1839x<sup>-1.07</sup></i>.
    </figcaption>
  </center>
</figure>



### Conclusion



Our simulations based on Simon's text generation model corroborate the observation that word frequency distributions often follows a power law. 

This power law distribution emerges from the interplay between the introduction of new words and the reinforcement of existing words. Specifically, generating new words tends to diminish the frequency disparity among words, while generating existing words tends to elongate the tail of the distribution. A delicate balance between these factors is crucial for maintaining a power law distribution; otherwise, deviations can arise, as evidenced in simulation with $\alpha=0.025$.



Ultimatley, the power law distribution encapsulates the notion of "the rich get richer", highlighting how word frequencies evolve within a text. 



### References

Simon, H. A. (1955). On a Class of Skew Distribution Functions. *Biometrika*, 42(3/4), 425-440. https://www.jstor.org/stable/2333389?origin=JSTOR-pdf



