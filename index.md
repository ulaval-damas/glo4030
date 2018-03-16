---
layout: home
title: Accueil GLO-4030/7030
---

<meta name="keywords" content="Deep Learning, course, tutorials, CNN"/>

## Forum de discussions

[piazza.com/ulaval.ca/winter2018/glo40307030/home](https://piazza.com/ulaval.ca/winter2018/glo40307030/home)

## Horaire

| Jour     | Heure               | Local    |
|----------|---------------------|----------|
| Mardi    | 12h30 à 15h20       | PLT-2341 |
| Vendredi | 10h30 à 11h30       | PLT-3928 |

## Ressource Jupyter de Calcul-Québec (pour GPU)

[https://jupyter.calculquebec.ca/](https://jupyter.calculquebec.ca/)

Pendant la période de laboratoire, nous avons une priorité d'accès, sous la réservation ``glo4030``. En dehors de la période de laboratoire, votre requête de notebook passera par une file d'attente.

## Travaux pratiques
- <a href="http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/TP1_remise.zip" target="_blank">TP1</a>, remise 9 février 2018, 23h55. Sans pénalité de retard jusqu'au 11 février, 23h55.
- <a href="http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/TP2.remise.zip" target="_blank">TP2</a>, remise 11 mars 2018, 23h55. 

## Choix de l'article à présenter oralement (GLO-7030 seulement)

Pour la présentation orale de 15 minutes, vous devez choisir un article en lien avec le cours et publié **après le 1er juin 2017**, sur le site d'[arxiv.org](https://arxiv.org) ou toute conférence respectable (CVPR, NIPS, RSS, ICCV, ECCV). Une manière simple d'en trouver un est de régulièrement consulter le site [arxiv-sanity](http://www.arxiv-sanity.com/), qui montre les articles les plus récents (dans la dernière semaine, essentiellement) triés selon les sujets, ou même le *hype*. Entre un *reddit* et un *slashdot*, faites-y un tour ;). L'horaire des présentations est ici-bas (il vous faudra dérouler probablement).

<iframe src="https://docs.google.com/spreadsheets/d/e/2PACX-1vTnYDfcPNLtDa9jsUvpOHuilBx3zL832-b54grHpY4N_5TffQjIT88IcOW6n4vgxbnpelLPXbDJX4Ll/pubhtml?gid=0&amp;single=true&amp;range=A1:E33&amp;headers=false&amp;widget=false&amp;headers=false&amp;chrome=false" width="800" height="600"></iframe>


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
   - [Cours sur Fast.ai](http://forums.fast.ai/t/another-treat-early-access-to-intro-to-machine-learning-videos/6826?source_topic_id=9398)
   
   

   
   
   
- Semaine 1 (16 janvier) : Plan de cours, introduction, réseau linéaire, fonctions d'activation

  **Lectures dans le manuel :** Chapitre 1, 6
  
  Nous verrons notamment quelles sont les innovations des 10 dernières années qui expliquent la résurgence des réseaux de neurones, en particulier l'apparition des réseaux profonds. Les premiers détails sur les réseaux en aval (feedforward) seront abordés.
  
  **Contenu détaillé :**
  - Introduction
  - Réseau feedforward (aval) de base, activations
  - Classificateur linéaire
  - Importance de la dérivabilité
  
  **Vidéos :**
  - [Feedforward neural network - artificial neuron](https://youtu.be/SGZ6BttHMPw?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Feedforward neural network - activation function](https://youtu.be/tCHIkgWZLOQ?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  
  **Acétates :**  
  - [01-Introduction.pdf](http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/01-Introduction.pdf)
  
  **Laboratoire :** Introduction à Pytorch et premiers essais sur des données standards (MNIST, CIFAR-10)
  - [Laboratoire 1](https://github.com/ulaval-damas/glo4030-labs/blob/master/Laboratoire%201.ipynb)


  
  
- Semaine 2 (23 janvier) : Feedforward, Fonctions de perte, Graphes de calcul, Backprop

  **Lectures dans le manuel :** Chapitre 6 

  **Acétates :**
  - <a href="http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/02-FeedForward-Loss-Graph-Backprop.pdf" target="_blank">02-FeedForward-Loss-Graph-Backprop.pdf</a>

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
   



- Semaine 3 (30 janvier) : Initialisation et optimisation 
  
  **Lectures dans le manuel :** Chapitre 8

  **Acétates :**   
  - <a href="http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/03-Deep_Learning_Optimization.pdf" target="_blank">03-Deep_Learning_Optimization.pdf</a>.
  
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




- Semaine 4 (6 février) : Régularisation
  
  **Lectures dans le manuel :** Chapitre 7
    
  **Vidéos :** 
  - [Training neural networks - regularization](https://youtu.be/JfkbyODyujw?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Deep learning - difficulty of training](https://youtu.be/YoiUlN_77LU?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Deep learning - unsupervised pre-training](https://youtu.be/Oq38pINmddk?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Deep learning - example](https://youtu.be/SXnG-lQ7RJo?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Deep learning - dropout](https://youtu.be/UcKPdAM8cnI?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  
  **Acétates :** <a href="http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/04-Regularisation.pdf" target="_blank">04-Regularisation.pdf</a>
  
  **Laboratoire :** 
  - [Laboratoire 4](https://github.com/ulaval-damas/glo4030-labs/blob/master/Laboratoire%204.ipynb)




- Semaine 5 (13 février) : Réseaux à convolution I (CNN) 
  
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


- Semaine 6 (20 février) : Réseaux à convolution II (CNN) 
  
  **Lectures dans le manuel :** Chapitre 9
  
  **Autres lectures :**
  - [K. He et al. Deep Residual Learning for Image Recognition, 2016.](https://arxiv.org/pdf/1512.03385.pdf)
  - <a href="https://arxiv.org/pdf/1602.07261.pdf" target="_blank">Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning</a>
  - <a href="https://arxiv.org/abs/1605.06431" target="_blank">Residual Networks Behave Like Ensembles of Relatively Shallow Networks</a>
   
  
  **Acétates :**
  - <a href="http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/06-CNN.pdf" target="_blank">06-CNN.pdf</a>
  
  **Laboratoire :** 
  - Pas de laboratoire cette semaine.




- Semaine 7 (27 février) : Examen Intra (GLO-4030 et GLO-7030) Local : PLT-2341



### Deuxième moitié

La deuxième moitié du cours est plus de type séminaire. On y fera un survol horizontal et rapide des concepts plus avancés. Les présentations orales y auront lieu, soit pendant une heure de cours, soit en remplacement des laboratoires.

{:.collapsible}

- Semaine 9 (13 mars) : Retour sur l'examen, début des présentations orales

  **Laboratoire :** 
  - Pas de laboratoire cette semaine.




- Semaine 10 (20 mars) : Spatial Transformer Network, Recursive Neural Networks (RNN)
  **Lectures dans le manuel :** Chapitre 10
  
  **Autres lectures :**
  - <a href="https://arxiv.org/abs/1506.02025" target="_blank">Spatial Transformer Networks</a>
  
  **Vidéos :**
  - <a href="https://youtu.be/6niqTuYFZLQ" target="_blank">cs231n Lecture 10 | Recurrent Neural Networks</a>
  
  **Laboratoire :** 
  - à venir
  
    
- Semaine 11 [TEMP] (27 mars) : Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU)


- Semaine 12 [TEMP] (3 avril) : Autoencodeurs, Word Embedding

  **Vidéos :**
  - [Autoencoder - definition](https://youtu.be/FzS3tMl4Nsc?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - loss function](https://youtu.be/xTU79Zs4XKY?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - example](https://youtu.be/6DO_jVbDP3I?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - linear autoencoder](https://youtu.be/xq-I0Rl8mt0?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - undercomplete vs. overcomplete hidden layer](https://youtu.be/5rLgoM2Pkso?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - denoising autoencoder](https://youtu.be/t2NQ_c5BFOc?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - contractive autoencoder](https://youtu.be/79sYlJ8Cvlc?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Deep learning - deep autoencoder](https://youtu.be/z5ZYm_wJ37c?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - <a href="https://youtu.be/ERibwqs9p38" target="_blank">Lecture 2 | Word Vector Representations: word2vec</a>


- Semaine 13 [TEMP] (10 avril) : Modèles d'attention

- Semaine 14 [TEMP] (17 avril) : Sujets avancés et expérimentaux

  **Laboratoire :** Présentations orales

- Semaine 15 [TEMP] (24 avril) : Modèles génératifs GAN 

  **Laboratoire :** Présentations orales

- Semaine 16 [TEMP] (1er mai) : Examen final (GLO-4030 SEULEMENT) 


En construction.

## Livre
Le livre (obligatoire) est **Deep Learning** par Goodfellow, Bengio et Courville.
Il est également disponible [en ligne](http://www.deeplearningbook.org).

<img src="https://mitpress.mit.edu/sites/default/files/9780262035613_0.jpg" width="250px">

## Liens utiles

#### Deep Learning Glossary

- <http://www.wildml.com/deep-learning-glossary/>

#### 100 Most Cited Deep Learning Papers (2012-2016)
- <https://github.com/terryum/awesome-deep-learning-papers>

#### Technos
Le framework utilisé dans le cours est PyTorch. Voici quelques liens utiles:

- <https://github.com/bharathgs/Awesome-pytorch-list>
- <https://github.com/aaron-xichen/pytorch-playground>

#### En savoir plus
- <http://cs231n.stanford.edu/index.html>
