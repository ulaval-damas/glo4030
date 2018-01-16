---
layout: home
title: Accueil GLO-4030/7030
---

## Forum de discussions

[piazza.com/ulaval.ca/winter2018/glo40307030/home](https://piazza.com/ulaval.ca/winter2018/glo40307030/home)

## Horaire

| Jour     | Heure               | Local    |
|----------|---------------------|----------|
| Mardi    | 12h30 à 15h20       | PLT-2341 |
| Vendredi | 10h30 à 11h30       | PLT-3920 |

## Ressource Jupyter de Calcul-Québec (pour GPU)

[https://jupyter.calculquebec.ca/](https://jupyter.calculquebec.ca/)

Pendant la période de laboratoire, nous avons une priorité d'accès, sous la réservation ``glo4030``. En dehors de la période de laboratoire, votre requête de notebook passera par une file d'attente.

## Choix de l'article à présenter oralement (GLO-7030 seulement)

Pour la présentation orale de 15 minutes, vous devez choisir un article en lien avec le cours et publié **après le 1er juin 2017**, sur le site d'[arxiv.org](https://arxiv.org) ou toute conférence respectable (CVPR, NIPS, RSS, ICCV, ECCV). Une manière simple d'en trouver un est de régulièrement consulter le site [arxiv-sanity](http://www.arxiv-sanity.com/), qui montre les articles les plus récents (dans la dernière semaine, essentiellement) triés selon les sujets, ou même le *hype*. Entre un *reddit* et un *slashdot*, faites-y un tour ;). L'horaire des présentations sera disponible bientôt.

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
   - [Coursera : Machine Learning](https://www.class-central.com/mooc/835/coursera-machine-learning)
   
   

   
   
   
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
  
  
  
  
  
  
- Semaine 2 [TEMP] (23 janvier) : Graphes de calculs, backprop

  **Lectures dans le manuel :** Chapitre 6 

  **Acétates :** à venir.

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
  
  **Laboratoire :** 
   



- Semaine 3 [TEMP] (30 janvier) : Initialisation et optimisation 
  
  **Lectures dans le manuel :** Chapitre 8

  **Acétates :** à venir.
  
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
  
  **Laboratoire :** 





- Semaine 4 [TEMP] (6 février) : Régularisation
  
  **Lectures dans le manuel :** Chapitre 7
  
  **Vidéos :** 
  - [Training neural networks - regularization](https://youtu.be/JfkbyODyujw?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Deep learning - difficulty of training](https://youtu.be/YoiUlN_77LU?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Deep learning - unsupervised pre-training](https://youtu.be/Oq38pINmddk?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Deep learning - example](https://youtu.be/SXnG-lQ7RJo?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Deep learning - dropout](https://youtu.be/UcKPdAM8cnI?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  
  **Acétates :** à venir.
  
  **Laboratoire :** 





- Semaine 5 [TEMP] (13 février) : Réseaux à convolution I (CNN) 
  
  **Lectures dans le manuel :** Chapitre 9
  
  **Autre lecture :** cs231n (Stanford) : [Convolutional networks](http://cs231n.github.io/convolutional-networks/)
     
  **Acétates :** à venir.
  
  **Laboratoire :** 





- Semaine 6 [TEMP] (20 février) : Réseaux à convolution II (CNN) 
  
  **Lectures dans le manuel :** Chapitre 9
  
  **Autres lectures :**
  - [K. He et al. Deep Residual Learning for Image Recognition, 2016.](https://arxiv.org/pdf/1512.03385.pdf)
  
  **Acétates :** à venir.
  
  **Laboratoire :** 




- Semaine 7 [TEMP] (27 février) : Examen Intra (GLO-4030 et GLO-7030) 



### Deuxième moitié

La deuxième moitié du cours est plus de type séminaire. On y fera un survol horizontal et rapide des concepts plus avancés. Les présentations orales y auront lieu, soit pendant une heure de cours, soit en remplacement des laboratoires.

{:.collapsible}
- Semaine 9 [TEMP] (13 mars) : À venir 

- Semaine 10 [TEMP] (20 mars) : Autoencodeurs, Word Embedding

  **Vidéos :**
  - [Autoencoder - definition](https://youtu.be/FzS3tMl4Nsc?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - loss function](https://youtu.be/xTU79Zs4XKY?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - example](https://youtu.be/6DO_jVbDP3I?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - linear autoencoder](https://youtu.be/xq-I0Rl8mt0?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - undercomplete vs. overcomplete hidden layer](https://youtu.be/5rLgoM2Pkso?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - denoising autoencoder](https://youtu.be/t2NQ_c5BFOc?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - contractive autoencoder](https://youtu.be/79sYlJ8Cvlc?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Deep learning - deep autoencoder](https://youtu.be/z5ZYm_wJ37c?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)

- Semaine 11 [TEMP] (27 mars) : 

- Semaine 12 [TEMP] (3 avril) : À venir 

- Semaine 13 [TEMP] (10 avril) : Autoencodeur



  **Laboratoire :** Présentations orales

- Semaine 14 [TEMP] (17 avril) : À venir 

  **Laboratoire :** Présentations orales

- Semaine 15 [TEMP] (24 avril) : À venir 

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
