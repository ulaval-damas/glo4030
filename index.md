---
layout: home
title: Accueil GLO-4030/7030 Hiver 2020
---

<meta name="keywords" content="Deep Learning, course, tutorials, CNN"/>

## COVID-19
Notez que l'horaire a passablement changé. Lisez vos courriels régulièrement!

## Horaire

| Jour     | Heure               | Local    |
|----------|---------------------|----------|
| Mardi    | 12h30 à 15h20       | PLT-2700  |
| Vendredi |  8h30 à 10h20       | PLT-3920  |

## Ressource Jupyter de Calcul-Québec (pour GPU)

[https://jupyterhub.helios.calculquebec.ca/](https://jupyterhub.helios.calculquebec.ca/)

## Format du rapport de projet

Le format du rapport, ainsi que le barème de correction, est disponible [ici]()


## Laboratoires

Pendant la période des laboratoires, nous avons une priorité d'accès, sous la réservation ``GLO7030``. En dehors de la période de laboratoire, votre requête de notebook passera par une file d'attente. Notez qu'à partir de maintenant, les laboratoires se feront à distance.

[Site web des laboratoires](https://github.com/ulaval-damas/glo4030-labs)

## Exercices non-évalués
- Quelques exercices sur les CNN [ExercicesCNN.pdf](http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/2020/ExercicesCNN.pdf). Le solutionnaire n'est pas disponibile pour l'instant, par manque de temps.
- Deuxième série d'exercices [ExercicesII.pdf](http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/2020/ExercicesII.pdf)


## Travaux pratiques

- [TP1](http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/2020/TP1_remise.v.1.01.2020.zip) : remise 14 février 2020 à 23h59.
- [TP2](assets/tps/TP2.2020.zip) : remise 10 avril 2020, 23h59.

## Jeux de données pour idées de projet
- **Nouveau!** Projet de détection de Cerfs dans des images (Biologie) [Description](http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/2020/DétectionCerfsImages.pdf)
- **Nouveau!** Projet de cartographie assistée du Ministère des Forêts, de la Faune et des Parcs! [Description](http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/2020/MFFP-CartographieAssistee.pdf)
- [datalist.com](https://www.datasetlist.com/)
- [paperswithcode.com](https://paperswithcode.com/sota)
- [awesome-public-datasets](https://github.com/awesomedata/awesome-public-datasets)

## Présentations orales de l’article (GLO-7030)
Au lieu d’une présentation de 10 minutes en direct, vous allez faire une capsule vidéo narrée de 10 minutes. Vous pouvez utiliser les fonctionalités de Powerpoint https://www.enseigner.ulaval.ca/ressources-pedagogiques/capsules-narrees . Vous pouvez aussi faire une capture d’écran + son directement de votre PC avec Screen-O-matic https://www.ene.ulaval.ca/captation-numerique . Si l’Université ne possède pas suffisamment de licences, n’hésitez pas à trouver d’autres solutions gratuites en ligne (et les indiquer sur le forum COVID-19). Ne vous inquiétez pas si ces solutions gratuites ajoutent des watermark/filigranes : vous ne serez pas pénalisés. Vous mettrez cette vidéo sur youtube, et j’ajouterai le lien sur le site du cours pour que les autres puissent les regarder. Les dates butoirs pour les remises de ces vidéos sont déplacées d’une semaine par rapport au calendrier original.  

L'horaire des présentations est disponible [au bas de la page web](#oraux-articles).

## Présentation de votre projet (GLO-7030 seulement)

Cette activité est annulée.

<!-- <iframe src="https://docs.google.com/spreadsheets/d/e/2PACX-1vTnYDfcPNLtDa9jsUvpOHuilBx3zL832-b54grHpY4N_5TffQjIT88IcOW6n4vgxbnpelLPXbDJX4Ll/pubhtml?gid=0&amp;single=true&amp;range=A1:E33&amp;headers=false&amp;widget=false&amp;headers=false&amp;chrome=false" width="800" height="600"></iframe> -->


## Support vidéo

Pour accélérer les leçons, des vidéos produite entre autre par Hugo Larochelle (Google Brain) sont disponibles. Je vous recommande fortement de les visionner avant la leçon. Nous le remercions d'ailleurs pour son autorisation d'utiliser ce matériel d'enseignement. 

## Plan de cours

### Première moitié

Cette première moitié du cours introduit les connaissances nécessaires pour concevoir et entraîner des réseaux profonds, particulièrement dans un contexte de reconnaissance d'images.

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
   
   
      
   
- Semaine 1 (14 janvier) : Plan de cours, introduction, réseau linéaire, fonctions d'activation

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
  - [01-Introduction-2020.pdf](http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/2020/01-Introduction-2020.pdf)
  
  **Laboratoire :** Introduction à Pytorch et premiers essais sur des données standards (MNIST, CIFAR-10)
  - [Laboratoire 1](https://github.com/ulaval-damas/glo4030-labs/blob/master/Laboratoire%201.ipynb)

 
  
- Semaine 2 (21 janvier) : Feedforward, Fonctions de perte, Graphes de calcul, Backprop

  **Lectures dans le manuel :** Chapitre 6 

  **Acétates :**
  - [02-FeedForward-Loss-Graph-Backprop.pdf](http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/2020/02-FeedForward-Loss-Graph-Backprop.pdf) 

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
   



- Semaine 3 (28 janvier) : Initialisation et optimisation 
  
  **Lectures dans le manuel :** Chapitre 8

  **Acétates :**   
  - [03-Optimisation.pdf](http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/2020/03-Optimisation.pdf) 
  
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




- Semaine 4 (4 février) : Régularisation
  
  **Lectures dans le manuel :** Chapitre 7
  
  **Acétates :** 
  - [04-Regularisation-2020.pdf](http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/2020/04-Regularisation-2020.pdf) 
  
  **Vidéos :** 
  - [Training neural networks - regularization](https://youtu.be/JfkbyODyujw?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Deep learning - difficulty of training](https://youtu.be/YoiUlN_77LU?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Deep learning - unsupervised pre-training](https://youtu.be/Oq38pINmddk?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Deep learning - example](https://youtu.be/SXnG-lQ7RJo?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Deep learning - dropout](https://youtu.be/UcKPdAM8cnI?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)

  **Laboratoire :** 
  - [Laboratoire 4](https://github.com/ulaval-damas/glo4030-labs/blob/master/Laboratoire%204.ipynb)



- Semaine 5 (11 février) : Réseaux à convolution I (CNN) 
  
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
  - [05-CNN-2020.pdf](http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/2020/05-CNN-2020.pdf) 
  
  **Laboratoire :** 
  - [Laboratoire 5](https://github.com/ulaval-damas/glo4030-labs/blob/master/Laboratoire%205.ipynb)
  - [Solution du labo 5](assets/notebooks/Laboratoire5_Solution.ipynb)  


- Semaine 6 (18 février) : Réseaux à convolution II (CNN)
  
  **Lectures dans le manuel :** Chapitre 9
  
  **Autres lectures :**
  - [K. He et al. Deep Residual Learning for Image Recognition, 2016.](https://arxiv.org/pdf/1512.03385.pdf)
  - <a href="https://arxiv.org/pdf/1602.07261.pdf" target="_blank">Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning</a>
  - <a href="https://arxiv.org/abs/1605.06431" target="_blank">Residual Networks Behave Like Ensembles of Relatively Shallow Networks</a>
  
  **Vidéos :** 
  - [Depthwise separable convolution](https://www.youtube.com/watch?v=T7o3xvJLuHk)
   
  
  **Acétates :**
  - [06-CNN-2020.pdf](http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/2020/06-CNN-2020.pdf) 

  
- Semaine 7 (25 février) : Examen Intra (GLO-4030 et GLO-7030) Local : PLT-2573 pour GLO-7030 avec nom de famille débutant A-D. PLT-2700 pour le reste.


### Deuxième moitié

La deuxième moitié du cours portera majoritairement sur les modèle récurrents.
Nous allons également voir quelques concepts plus avancés comme les réseaux génératifs (GAN) et les modèles d'attention.

{:.collapsible}
- Semaine 9 (10 mars) : Retour sur l'examen, Recurrent Neural Networks (RNN)

  **Lectures dans le manuel :** Chapitre 10

  **Vidéos :**
  - <a href="https://youtu.be/6niqTuYFZLQ" target="_blank">cs231n Lecture 10 | Recurrent Neural Networks</a>
  - Lire le chapitre 10 avant de regarder : <a href="https://youtu.be/AYku9C9XoB8" target="_blank">Y. Bengio - RNN (DLSS 2017)</a>
  
  **Acétates :**
  - [07-RNN-2020.pdf](http://www2.ift.ulaval.ca/~pgiguere/cours/DeepLearning/2020/07-RNN-2020.pdf) 


  **Notebooks :** 
  - [Laboratoire](https://github.com/ulaval-damas/glo4030-labs/blob/master/Laboratoire%206.ipynb)
  - [Parameters](assets/notebooks/Parameters - Sol.ipynb)
  - [RNNs](assets/notebooks/RNNs - Sol.ipynb)
 
 
- Semaine 10 (17 mars) : Activités annulées

{:.collapsible}
- Semaine 10* (24 mars) : Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), Word Embeddings, Modèles de langue, DeepRNN

  **Lectures dans le manuel :** Chapitre 10

  **Vidéos narrées :**
  - [LSTM et GRU](https://youtu.be/RTYViLjrwCE)
  - [Word embeddings #1 - Introduction](https://youtu.be/lu02OULkhQE)
  - [Word embeddings #2 - Réseaux de neurones](https://youtu.be/sCHQ4DddL74)
  - [Word embeddings #3 - Embeddings de phrases](https://youtu.be/lB6XQEUOevU )
  - [Modèles de langue #1 - Introduction](https://youtu.be/Rch0x1FLP4c)
  - [Modèles de langue #2 - Deep RNNs](https://youtu.be/GQOPOjrSjMk)
  - [Modèles de langue #3 - ELMo](https://youtu.be/KUnsRa4L5OY)

  **Acétates :**
  - [07-LSTM.pdf](assets/slides/07-LSTM.pdf)
  - [08-WordEmbeddings.pdf](assets/slides/08-WordEmbeddings.pdf)
  - [08-ModèlesDeLangue.pdf](assets/slides/08-Mod%C3%A8lesDeLangue.pdf)

  **Autres lectures :**
  - <a href="http://colah.github.io/posts/2015-08-Understanding-LSTMs/" target="_blank">Understanding LSTM</a>
  - <a href="http://karpathy.github.io/2015/05/21/rnn-effectiveness/" target="_blank">The Unreasonable Effectiveness of Recurrent Neural Networks</a>
  - <a href="https://arxiv.org/pdf/1503.04069.pdf" target="_blank">LSTM: A Search Space Odyssey</a>
  - <a href="http://proceedings.mlr.press/v37/jozefowicz15.pdf" target="_blank">An Empirical Exploration of Recurrent Network Architectures</a>

  **Notebooks :**
  - [LSTM](assets/notebooks/LSTMs.ipynb)
  - [Sequence Classification](assets/notebooks/sequence_classification.ipynb)
  - [Embeddings](assets/notebooks/Embeddings.ipynb)

  **Laboratoire :** 
  - RNN et LSTM

{:.collapsible}
- Semaine 11* (31 mars) : Autoencodeurs, Seq2Seq, Modèles d'attention 

  **Lectures dans le manuel :** Section 10.12 Explicit Memory
  
  
  **Vidéos narrées :**
  - [Autoencodeurs](https://youtu.be/ILBFADVFE5M) 
  - [Spatial Transformer Network](https://youtu.be/awrgZAdU964)
  - [3.1-Transformer-Decoder](https://youtu.be/PIkrddD4Jd4)
  - [3.2-GPT](https://youtu.be/F4tmxXyiVo0)
  - [3.3-GPT-2](https://youtu.be/xmWln3ctNhQ)
  - [04-01 RNN and Attention](https://youtu.be/wz2WRq93a7w)
  - [04-02 Attention Is All You Need](https://youtu.be/i3EIF5QFnNw)
  

  **Autres lectures :**
  - <a href="https://arxiv.org/abs/1409.0473" target="_blank">Neural Machine Translation by Jointly Learning to Align and Translate</a>
  - <a href="https://arxiv.org/abs/1706.03762" target="_blank">Attention Is All You Need</a>
  - <a href="https://youtu.be/ERibwqs9p38" target="_blank">Lecture 2 | Word Vector Representations: word2vec</a>
  - <a href="https://arxiv.org/pdf/1411.2738.pdf" target="_blank">word2vec Parameter Learning Explained</a>
  

  **Acétates :**
  - [08-Autoencodeur.pdf](assets/slides/08-Autoencodeur.pdf)
  - [09-STN.pdf](assets/slides/09-STN.pdf)
  - [3.0-Transformers-Timeline.pdf](assets/slides/3.0-Transformers-Timeline.pdf)
  - [3.1-Transformer-Decoder.pdf](assets/slides/3.1-Transformer-Decoder.pdf)
  - [3.2-GPT.pdf](assets/slides/3.2-GPT.pdf)
  - [3.3-GPT-2.pdf](assets/slides/3.3-GPT-2.pdf)
  - [04.01-RNNandAttention.pdf](assets/slides/04.01-RNNandAttention.pdf)
  - [04.02-AttentionIsAllYouNeed.pdf](assets/slides/04.02-AttentionIsAllYouNeed.pdf)
  

  **Laboratoire** 
  - [Labo 7](https://github.com/ulaval-damas/glo4030-labs/blob/master/Laboratoire%207.ipynb)
  

  **Notebooks**
  - [Attention Simple](assets/notebooks/tutoriel-attn.ipynb)
  
  
  **Autres liens pertinents :**
  - <a href="http://nlp.seas.harvard.edu/2018/04/03/attention.html" target="_blank">Code pour Attention Is All You Need</a>
  - <a href="https://sacred.readthedocs.io/en/latest/" target="_blank">Librairie d'expérimentations</a>
  - <a href="https://github.com/vivekratnavel/omniboard" target="_blank">Visualisation des expérimentations</a>
  - <a href="https://medium.com/@hadyelsahar/writing-code-for-natural-language-processing-research-emnlp2018-nlproc-a87367cc5146" target="_blank">Écrire du code pour la recherche</a>
  - [Autoencoder - definition](https://youtu.be/FzS3tMl4Nsc?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - loss function](https://youtu.be/xTU79Zs4XKY?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - example](https://youtu.be/6DO_jVbDP3I?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - linear autoencoder](https://youtu.be/xq-I0Rl8mt0?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - undercomplete vs. overcomplete hidden layer](https://youtu.be/5rLgoM2Pkso?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - denoising autoencoder](https://youtu.be/t2NQ_c5BFOc?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Autoencoder - contractive autoencoder](https://youtu.be/79sYlJ8Cvlc?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - [Deep learning - deep autoencoder](https://youtu.be/z5ZYm_wJ37c?list=PL6Xpj9I5qXYEcOhn7TqghAJ6NAPrNmUBH)
  - <a href="https://youtu.be/5WoItGTWV54?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&t=2995" target="_blank">Lecture 13 | Generative Models (partie GAN)</a>  


{:.collapsible}
- Semaine 12* (7 avril) :  Détection d'objets, Segmentation d'image

  **Vidéos narrées :**
  - Vidéos à venir
  
  
  **Acétates :**
  - À venir
  

{:.collapsible}
- Semaine 13* (14 avril) : Modèles GAN (Generative Adverserial Network)

  **Acétates :**
  - [11-GANs.pdf](assets/slides/11-GANs.pdf)
  
  
  **Vidéos :**
  - [GAN (Vidéo du cours de 2018!)](https://www.youtube.com/watch?v=Lze_9nZrh5E)
  

  **Autres Liens :**
  - <a href="https://github.com/soumith/ganhacks" target="_blank">Trucs et astuces pour entrainer des GANs</a> 
  

{:.collapsible}
- Semaine 15 (21 avril) : Examen final (GLO-4030 SEULEMENT, 12h30 à 14h20, modalité à déterminer) 

L'examen se fera en ligne. Les détails suivront.

  **Révision :**
  - RNN/LSTM
  - Word Embeddings
  - Modèles de langue (ELMo)
  - Attention (Bahdanau, Attention is all you need)
  - AutoEncoder
  - GAN
  

## Oraux articles
Horaire des remises des vidéos narrés d'un article (GLO-7030)
 <iframe src="https://docs.google.com/spreadsheets/d/e/2PACX-1vRj4ZKK-hoIRwsCNTzAz8adJr5aFS4KCInby_4aRCtwDwZP-_eg0VG76kzHG3ZwbY65fCa3tKxtjaY5/pubhtml?gid=0&amp;single=true&amp;widget=true&amp;headers=false&amp;range=A1:F80" width="940" height="1000"></iframe>

## Livre
Le livre (obligatoire) est **Deep Learning** par Goodfellow, Bengio et Courville.
Il est également disponible [en ligne](http://www.deeplearningbook.org).

<img src="https://paulvanderlaken.files.wordpress.com/2017/10/9780262035613_0.jpg" width="250px">


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
