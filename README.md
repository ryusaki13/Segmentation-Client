# Segmentation Client

Dans un environnement fortement concurrentiel, les entreprises cherchent constamment à augmenter leur part de marché, leur chiffre d'affaires et leur marge. Cela passe par des offres commerciales et des campagnes marketing ciblées, adaptées à chaque groupe de clients en fonction de leurs caractéristiques communes et intrinsèques.  

Ainsi, il est essentiel de segmenter les clients pour mieux comprendre leurs comportements et optimiser les actions marketing.  

Ce projet vise à réaliser une **segmentation client** sur les données d'une entreprise de retail entre 2010 et 2012.

## Outils et Méthodologie

- Prétraitement des données (Data preprocessing)  
- Analyse RFM (Récence, Fréquence, Montant)  
- Analyse en Composantes Principales (ACP)  
- Classification non supervisée : K-means  
- Création et export de pipeline pour réutilisation  

## Résultats

| Cluster                             | Description du segment                                                                                       |
|-------------------------------------|--------------------------------------------------------------------------------------------------------------|
| Cluster 0 : Inactifs                | Clients qui n'ont pas acheté depuis longtemps ; faible fréquence et faible montant dépensé                   |
| Cluster 1 : Actifs de classe moyenne| Clients fidèles depuis longtemps ; dépensent beaucoup et achètent très fréquemment                           |
| Cluster 2 : VIP & Champions         | Achat régulier et montant moyen ; fréquence et récence correctes ; possibilité de montée en gamme           |
| Cluster 3 : Anciens peu engagés     | Clients qui ont acheté très récemment ; faible fréquence et montant modeste ; reviennent après une longue pause |
