---

title: "Effect of Attention on Text Classification Performance"
date: 2024-11-09
typora-root-url: ./..
---



In this post, we continue from our [previous study](https://kezhaozhang.github.io/2024/10/31/rnn.html) on text classification using a recurrent neural network (RNN) model.  In that study, we explored how different Byte-Pair Encoding (BPE) settings and RNN architectures impact classification performance. We found that BPE configurations significantly influenced the classification performances. This follow-up study focuses on understanding the effect of incorporating an attention mechanism into the RNN model. 



### Problem Setup and Model Architecture



The task is to classify sentences from Chinese Wikipedia articles into three topics: 物理学 (physics), 数学 (mathematics), and 生物学 (biology). Three articles corresponding to these topics are used for training, and three additional articles on 万有引力 (gravity), 群论 (group theory), and 细胞 (cell) are used as the validation set. Each article is split into sentences using the Chinese period "。".



The RNN architecture from the previous study is depicted in Figure 1.



<figure>
  <center>
  <img src="/assets/images/rnn.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 1. RNN architecture for text topic classification.
    </figcaption>
  </center>
</figure>



In this study, we add an attention layer before the LSTM layer, as shown in Figure 2.  The attention mechanism allows the model to focus on different parts of the input sentences, potentially improving classification accuracy.   In the attention layer, the dimensions of the Key, Query, and Value vectors are equal, and we denote this common dimension by $M$. 

<figure>
  <center>
  <img src="/assets/images/attention_attendtion_net_graphic.svg" width="750">
   </center>
  <center>
    <figcaption> Figure 2. RNN architecture with an attention layer before the LSTM layer.
    </figcaption>
  </center>
</figure>



### Experimental Setup and Result



In this study, we focus on two key parameters of the attention mechanism: 

1. $M$, the dimension of the Key, Query, and Value vectors.
2.  $p$, the dropout rate for the attention weights. 

We tested different combinations of these parameters to assess their impact on the classification performance. 



Figure 3 shows the validation set error rate for various values of $M$ (the attention weight dimension) and $p$ (dropout rate). The dashed line is the best error rate from the baseline RNN model without the attention layer, which was reported in the previous study. 



<figure>
  <center>
  <img src="/assets/images/attention_errorate_vs_p_for_various_M.svg" width="600">
   </center>
  <center>
    <figcaption> Figure 3. Validation set error rate for different dropout rates (<i>p</i>) and attention weight dimensions (<i>M</i>).
    </figcaption>
  </center>
</figure>



Incorporating the attention layer ($M=50$ and $p=0.2$ or $p=0.55$) reduces the error rate from 0.103 (without attention) to 0.086. While the improvement is modest, the attention mechanism helps enhance the model's performance. 



The attention weight matrices for the trained networks with $M=50$ and $p=0.2$ are visualized in Figure 4.

<figure>
  <center>
  <img src="/assets/images/attention_weight_matrixplot.svg" width="700">
   </center>
  <center>
    <figcaption> Figure 4. Attention weight matrix plot for <i>M=50</i> and <i>p=0.2</i>.
    </figcaption>
  </center>
</figure>



### Visualizing Attention Weights



To better understrand how the attention mechnaism works, we visualize the attention weights for a single sample from the validation set.  The sentence we use is from the 万有引力 (gravity) article: 

"根據牛頓第三運動定律，地球同时也受到下落的物体等值反向的力的作用，意味着地球也将加速向物体运动" (According to Newton's third law of motion, the earth is also acted upon by an equal and opposite force from the falling object, which means that the earth will also accelerate towards the object.) 



We tokenize this sentence using the [BPEmb](https://bpemb.h-its.org/) encoder,  and the attention weights are computed using both the BPEmb embedding and the attention weight matrics from the trained network. Figure 5 shows the attention weights between the tokens, with the tokens in the left column representing the queries and those in the right columns representing the keys. Darker lines indicate stronger attention weights between the tokens. 



<figure>
  <center>
  <img src="/assets/images/attention_weight_line_plot.svg" width="400">
   </center>
  <center>
    <figcaption> Figure 5. Visualization of attention weights for one validation sentence. The opacity of the lines indicates the strength of attention between tokens. Tokens on the left are queries, and those on the right are keys. 
    </figcaption>
  </center>
</figure>



The attention weights vary significantly across token pairs, demonstrating that the model is able to focus on different parts of the sentence depending on the context. This variablity in attention allows the model to prioritize certain relationship between words, ehivh likely contributes to its improved performance. 



### Conclusion



Incoporating the attention mechanism into the RNN architecture leads to a modest but signifcant improvement in text classification performance. While the gain is smaller compared to other factors like word embeddings, the attention mechanism plays a crucial role in helping the model focus on the most relevant parts of the input sequence. By visualizng the attention weights, we can confirm that the model is making use of these highlighted relationships between tokens to improve its understanding of the input text. 



Future studies could explore further optimizations to the attention mechanism, such as experimenting with different attention types (e.g., multi-head attention) or fine-tuning the attention parameters. to achieve even greater improvements.

