---
layout: home
title: Accueil GLO-4030/7030
---

<meta name="keywords" content="Deep Learning, course, tutorials, CNN"/>

## Horaire

| Jour     | Heure               | Local    |
|----------|---------------------|----------|
| Mardi    | 12h30 à 15h20       | PLT-2783 |
| Vendredi | 9h30 à 10h30        | PLT-3920 |

## Ressource Jupyter de Calcul-Québec (pour GPU)

[https://jupyter.calculquebec.ca/](https://jupyter.calculquebec.ca/)

Pendant la période de laboratoire, nous avons une priorité d'accès, sous la réservation ``glo4030``. En dehors de la période de laboratoire, votre requête de notebook passera par une file d'attente.

[Site web des laboratoires](https://github.com/ulaval-damas/glo4030-labs)

## Travaux pratiques

- [TP1](assets/tps/TP1_remise.zip), remise 8 février 2019, 23h59.

## Présentation de votre projet (GLO-7030 seulement)

Pour la présentation de votre projet, vous devez préparer et présenter une affiche.
Un [template](https://github.com/GRAAL-Research/graal-poster) vous est mis à disposition et il vous est fortement recommandé de l'utiliser pour vous faciliter le travail.

<!-- <iframe src="https://docs.google.com/spreadsheets/d/e/2PACX-1vTnYDfcPNLtDa9jsUvpOHuilBx3zL832-b54grHpY4N_5TffQjIT88IcOW6n4vgxbnpelLPXbDJX4Ll/pubhtml?gid=0&amp;single=true&amp;range=A1:E33&amp;headers=false&amp;widget=false&amp;headers=false&amp;chrome=false" width="800" height="600"></iframe> -->


## Support vidéo

Pour accélérer les leçons, des vidéos produite entre autre par Hugo Larochelle (Google Brain) sont disponibles. Je vous recommande fortement de les visionner avant la leçon. Nous le remercions d'ailleurs pour son autorisation d'utiliser ce matériel d'enseignement. 

## Plan de cours

### Première moitié

Cette première moitié du cours introduit les connaissances nécessaires pour concevoir et entraîner des réseaux profonds, particulièrement dans un contexte de reconnaissance d'images.



[TEMP] indique que l'entrée de semaine est en cours de rédaction.

{:.collapsible}
- Semaine 0 : Mise à niveau : apprentissage automatique, probabilités, etc.

   Nous prenons pour acquis que vous maîtrisez la plupart des concepts de base de l'apprentissage automatique, d'algèbre linéaire, et de probabilités. Pour ceux qui ont fait les préalables, vous devriez déjà avoir ces bases. Pour les autres, vous devrez déployer des efforts supplémentaires pour acquérir ces fondements, et ce sur votre propre temps. Nous ne pouvons malheureusement pas utiliser du temps en classe pour expliquer ce que sont les concepts fondamentaux comme :
   - l'apprentissage supervisé, et non-supervisé;
   - le surapprentissage (overfit);
   - la classification vs. la régression;
   - les classificateurs classiques comme k-NN, SVM, fonction logistique;
   - une idée approximative de qu'est-ce que la complexité (puissance) d'un classificateur.
   - ensembles d'entraînement, de validation, et de test;
   - la validation croisée;
   - les méthodes de réduction de la dimensionalité (PCA);
   - les priors.  
   
   **Lectures dans le manuel :** Chapitre 2, 3, 4.
   
   Liens pour l'apprentissage automatique :
   - [Vidéo](http://videolectures.net/deeplearning2016_precup_machine_learning/) résumant le machine learning par Doina Precup
   - [Udacity : Introduction to Machine Learning](https://www.udacity.com/course/intro-to-machine-learning--ud120)
   - <a href="https://www.class-central.com/mooc/835/coursera-machine-learning" target="_blank">Coursera : Machine Learning</a>
   - [Cours sur Fast.ai](https://course.fast.ai/ml)
   
   

      
   
- Semaine 1 (15 janvier) : Plan de cours, introduction, réseau linéaire, fonctions d'activation

  **Lectures dans le manuel :** Chapitre 1, 6
  
  Nous verrons notamment quelles sont les innovations des 10 dernières années qui expliquent la résurgence des réseaux de neurones, en particulier l'apparition des réseaux profonds.
  Les premiers détails sur les réseaux en aval (feedforward) seront abordés.
  
  **Contenu détaillé :**
  - Introduction
  - Réseau feedforward (aval) de base, activations
  - Classificateur linéaire
  - Importance de la dérivabilité
  
  **Vidéos :**
  - [Feedforward neural network - artificial neuron](https://youtu.be/SGZ6BttHMPw?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Feedforward neural network - activation function](https://youtu.be/tCHIkgWZLOQ?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  
  **Acétates :**  
  - [Introduction](assets/slides/01-intro.pdf)
  
  **Laboratoire :** Introduction à Pytorch et premiers essais sur des données standards (MNIST, CIFAR-10)
  - [Laboratoire 1](https://github.com/ulaval-damas/glo4030-labs/blob/master/Laboratoire%201.ipynb)


  
  
- Semaine 2 (22 janvier) : Feedforward, Fonctions de perte, Graphes de calcul, Backprop

  **Lectures dans le manuel :** Chapitre 6 

  **Acétates :**
  - [Fonctions de pertes et graphe de calcul](assets/slides/02-loss-graph.pdf)

  **Vidéos :**
  - [Training neural networks - empirical risk minimization](https://youtu.be/5adNQvSlF50?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Training neural networks - loss function](https://youtu.be/PpFTODTztsU?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Training neural networks - output layer gradient](https://youtu.be/1N837i4s1T8?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Training neural networks - hidden layer gradient](https://youtu.be/xFhM_Kwqw48?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Training neural networks - activation function derivative](https://youtu.be/tf9p1xQbWNM?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Training neural networks - parameter gradient](https://youtu.be/p5tL2JqCRDo?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Training neural networks - backpropagation](https://youtu.be/_KoWTD8T45Q?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Computation Graph (Andrew Ng)](https://youtu.be/hCP1vGoCdYU?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)
  - [Derivatives With Computation Graphs (Andrew Ng)](https://youtu.be/nJyUyKN-XBQ?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)
  - [Logistic Regression Gradient Descent (Andrew Ng)](https://youtu.be/z_xiwjEdAC4?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)
  
  **Laboratoire :** Pytorch sous le capot,  (MNIST, CIFAR-10)
  - [Laboratoire 2](https://github.com/ulaval-damas/glo4030-labs/blob/master/Laboratoire%202.ipynb)
   



- Semaine 3 (29 janvier) : Initialisation et optimisation 
  
  **Lectures dans le manuel :** Chapitre 8

  **Acétates :**   
  - [Optimisation et initilisation](assets/slides/03-optim.pdf)
  
  **Vidéos :**
  - [Training neural networks - parameter initialization](https://youtu.be/sLfogkzFNfc?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Training neural networks - model selection](https://youtu.be/Fs-raHUnF2M?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Training neural networks - optimization](https://youtu.be/Bver7Ttgb9M?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Gradient Descent on m Examples (Andrew Ng)](https://youtu.be/KKfZLXcF-aE?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)
  - [Normalizing Inputs (Andrew Ng)](https://youtu.be/FDCfw-YqWTE?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc)
  - [BN: Normalizing Activations in a Network (Andrew Ng)](https://youtu.be/tNIpEZLv_eg?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc)
  - [BN: Fitting Batch Norm Into Neural Networks (Andrew Ng)](https://youtu.be/em6dfRxYkYU?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc)
  - [BN: Why Does Batch Norm Work? (Andrew Ng)](https://youtu.be/nUUqwaxLnWs?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc)
  - [BN: Batch Norm at Test Time (Andrew Ng)](https://youtu.be/5qefnAek8OA?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc)
  - <a href="https://youtu.be/lAq96T8FkTw?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc" target="_blank">Exponentially Weighted Averages (Andrew Ng)</a>
  - <a href="https://youtu.be/NxTFlzBjS-4?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc" target="_blank">Understanding Exponentially Weighted Averages (Andrew Ng)</a>
  - <a href="https://youtu.be/lWzo8CajF5s?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc" target="_blank">Bias Correction of Exponentially Weighted Averages (Andrew Ng)</a>
  
  **Laboratoire :** 
  - [Laboratoire 3](https://github.com/ulaval-damas/glo4030-labs/blob/master/Laboratoire%203.ipynb)
  <!-- - Laboratoire 3 (À venir) -->




- Semaine 4 (5 février) : Régularisation
  
  **Lectures dans le manuel :** Chapitre 7
  
  **Acétates :** <a href="http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/04-Regularisation.pdf" target="_blank">04-Regularisation.pdf</a>
  - [Régularisation](assets/slides/04-regularisation.pdf)
  
  **Vidéos :** 
  - [Training neural networks - regularization](https://youtu.be/JfkbyODyujw?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Deep learning - difficulty of training](https://youtu.be/YoiUlN_77LU?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Deep learning - unsupervised pre-training](https://youtu.be/Oq38pINmddk?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Deep learning - example](https://youtu.be/SXnG-lQ7RJo?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Deep learning - dropout](https://youtu.be/UcKPdAM8cnI?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)

  **Laboratoire :** 
  - [Laboratoire 4](https://github.com/ulaval-damas/glo4030-labs/blob/master/Laboratoire%204.ipynb)
  <!-- - Laboratoire 4 (À venir) -->




- Semaine 5 (12 février) : Réseaux à convolution I (CNN) 
  
  **Lectures dans le manuel :** Chapitre 9
  
  **Autres lectures :** 
  - cs231n (Stanford) : [Convolutional networks](http://cs231n.github.io/convolutional-networks/)
  - <a href="https://arxiv.org/pdf/1312.4400" target="_blank">Network in Network</a>
  - <a href="https://arxiv.org/pdf/1409.1556" target="_blank">Very Deep Convolutional Networks for Large-Scale Image Recognition</a>
  - <a href="https://arxiv.org/pdf/1409.4842" target="_blank">Going Deeper with Convolutions</a>
     
  **Vidéos :** 
  - <a href="https://youtu.be/bNb2fEVKeEo?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv" target="_blank">cs231n lesson 5</a>     
  - <a href="https://youtu.be/DAOcjicFr1Y?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv" target="_blank">cs231n Lecture 9 | CNN Architectures</a>
  
  **Acétates :**
  - <a href="http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/05-CNN.pdf" target="_blank">05-CNN.pdf</a>
  
  **Laboratoire :** 
  - [Laboratoire 5](https://github.com/ulaval-damas/glo4030-labs/blob/master/Laboratoire%205.ipynb)
  <!-- - Laboratoire 5 (À venir) -->


- Semaine 6 (19 février) : Réseaux à convolution II (CNN) et Spatial Transformer Network
  
  **Lectures dans le manuel :** Chapitre 9
  
  **Autres lectures :**
  - [K. He et al. Deep Residual Learning for Image Recognition, 2016.](https://arxiv.org/pdf/1512.03385.pdf)
  - <a href="https://arxiv.org/pdf/1602.07261.pdf" target="_blank">Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning</a>
  - <a href="https://arxiv.org/abs/1605.06431" target="_blank">Residual Networks Behave Like Ensembles of Relatively Shallow Networks</a>
  - <a href="https://arxiv.org/abs/1506.02025" target="_blank">Spatial Transformer Networks</a>
   
  
  **Acétates :**
  - <a href="http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/06-CNN.pdf" target="_blank">06-CNN.pdf</a>
  - <a href="http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/07-SpatialTransformNet.pdf" target="_blank">07-SpatialTransformNet.pdf</a>
  
  **Laboratoire :** 
  - Pas de laboratoire cette semaine.




- Semaine 7 (26 février) : Examen Intra (GLO-4030 et GLO-7030) Local : PLT-2783



### Deuxième moitié

La deuxième moitié du cours portera majoritairement sur les modèle récurrents.
Nous allons également voir quelques concepts plus avancés comme les réseaux génératifs (GAN) et les modèles d'attention.

{:.collapsible}

- Semaine 9 (12 mars) : Retour sur l'examen, Recurrent Neural Networks (RNN)

  **Lectures dans le manuel :** Chapitre 10

  **Vidéos :**
  - <a href="https://youtu.be/6niqTuYFZLQ" target="_blank">cs231n Lecture 10 | Recurrent Neural Networks</a>
  - Lire le chapitre 10 avant de regarder : <a href="https://youtu.be/AYku9C9XoB8" target="_blank">Y. Bengio - RNN (DLSS 2017)</a>
  
  **Acétates :**
  - <a href="http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/07-RNN-A.pdf" target="_blank">07-RNN-A.pdf</a>

  **Laboratoire :** 
  - Pas de laboratoire cette semaine.
  
    
- Semaine 10 (19 mars) : Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU)

  **Lectures dans le manuel :** Chapitre 10

  **Acétates :**
  - <a href="http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/07-RNN-B.pdf" target="_blank">07-RNN-B.pdf</a>
  - <a href="http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/07-LSTM.pdf" target="_blank">07-LSTM.pdf</a>

  **Autres lectures :**
  - <a href="http://colah.github.io/posts/2015-08-Understanding-LSTMs/" target="_blank">Understanding LSTM</a>
  - <a href="https://arxiv.org/pdf/1503.04069.pdf" target="_blank">LSTM: A Search Space Odyssey</a>
  - <a href="http://proceedings.mlr.press/v37/jozefowicz15.pdf" target="_blank">An Empirical Exploration of Recurrent Network Architectures</a>

  **Laboratoire :** 
  - RNN et LSTM


- Semaine 11 (26 mars) : Word Embeddings, DeepRNN, modèles de langue

  **Acétates :**
  - À venir


- Semaine 12 (2 avril) : Seq2Seq, Modèles d'attention

  **Lectures dans le manuel :** Section 10.12 Explicit Memory

  **Autres lectures :**
  - <a href="https://arxiv.org/abs/1409.0473" target="_blank">Neural Machine Translation by Jointly Learning to Align and Translate</a>
  - <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention Is All You Need</a>
  - <a href="https://youtu.be/ERibwqs9p38" target="_blank">Lecture 2 | Word Vector Representations: word2vec</a>
  - <a href="https://arxiv.org/pdf/1411.2738.pdf" target="_blank">word2vec Parameter Learning Explained</a>

  **Acétates :**
  - <a href="http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/09-Attention.pdf" target="_blank">09-Attention.pdf</a>

  **Laboratoire :** 
  - (À venir).

  **Autres liens pertinents :**
  - <a href="http://nlp.seas.harvard.edu/2018/04/03/attention.html" target="_blank">Code pour Attention Is All You Need</a>


- Semaine 13 (9 avril) : Autoencodeurs, Modèles GAN (Generative Adverserial Network)

  **Acétates :**
  - <a href="http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/08-Autoencoder.pdf" target="_blank">08-Autoencoder.pdf</a>
  - <a href="http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/11-GANs.pdf" target="_blank">11-GANs</a>

  **Autres Liens :**
  - <a href="https://github.com/soumith/ganhacks" target="_blank">Trucs et astuces pour entrainer des GANs</a>  
    
  **Vidéos :**
  - [Autoencoder - definition](https://youtu.be/FzS3tMl4Nsc?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - loss function](https://youtu.be/xTU79Zs4XKY?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - example](https://youtu.be/6DO_jVbDP3I?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - linear autoencoder](https://youtu.be/xq-I0Rl8mt0?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - undercomplete vs. overcomplete hidden layer](https://youtu.be/5rLgoM2Pkso?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - denoising autoencoder](https://youtu.be/t2NQ_c5BFOc?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - contractive autoencoder](https://youtu.be/79sYlJ8Cvlc?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Deep learning - deep autoencoder](https://youtu.be/z5ZYm_wJ37c?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - <a href="https://youtu.be/5WoItGTWV54?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&t=2995" target="_blank">Lecture 13 | Generative Models (partie GAN)</a>  


- Semaine 14 (16 avril) : Sujets avancés : Ladder Net, Réseaux Siamois et Distillation

  **Acétates :**
  - <a href="http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/10_SujetsSpeciaux.pdf" target="_blank">10_SujetsSpeciaux.pdf</a>
    
  **Vidéos :**
  - <a href="https://youtu.be/ZlyqNiPFu2s" target="_blank">Ladder Network</a>  
  - <a href="https://youtu.be/6jfw8MuKwpI?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF" target="_blank">Siamese Network (Andrew Ng)</a>
  - <a href="https://youtu.be/d2XB5-tuCWU?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF" target="_blank">Triplet loss (Andrew Ng)</a>

    
- Semaine 15 (23 avril) : Examen final (GLO-4030 SEULEMENT, PLT-2783 12h30 à 14h20) 


- Semaine 16 [TEMP] (30 avril) : Présentation des affiches, détails à venir. 


## Livre
Le livre (obligatoire) est **Deep Learning** par Goodfellow, Bengio et Courville.
Il est également disponible [en ligne](http://www.deeplearningbook.org).

<img src="https://paulvanderlaken.files.wordpress.com/2017/10/9780262035613_0.jpg" width="250px">

## Liens utiles

#### Deep Learning Glossary

- <http://www.wildml.com/deep-learning-glossary/>

#### 100 Most Cited Deep Learning Papers (2012-2016)
- <https://github.com/terryum/awesome-deep-learning-papers>

#### Jeux de données
- <https://github.com/awesomedata/awesome-public-datasets>

#### Technos
Le framework utilisé dans le cours est PyTorch. Voici quelques liens utiles:

- <https://github.com/bharathgs/Awesome-pytorch-list>
- <https://github.com/aaron-xichen/pytorch-playground>

#### En savoir plus
- <http://cs231n.stanford.edu/index.html>
