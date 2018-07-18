# Case #1: "bad-or-not-domain"
I've tried to guess if it's possible to distinguish good domains (like `google.com`), 
from bad one (like `atinyfishingwebsite.xxx`) using only domain name. Spoiler: I've failed.

Let me present the approach and results. To classify domain names I've used very simple recurrent network: 
layer of Gated Linear Units (hidden size = 32) ---> DropOut with p = 75%  ---> FC layer ---> Softmax and Cross-Entropy loss

Train regime:
a) ADAM solver
b) 100 epochs
c) LR = 0.001 devided by 2 every 40 epochs
d) Minibatch size = 32

![Losses and accuracies for 100 epochs](https://github.com/tinnulion/bad-domain-names/blob/master/bad_or_not_domain/results/plot.png)

It's not hard to understand that model overfits like crazy, having relatively few parameters at the same time.

I see few ways to improve the situation:
1. Try something really simple, like logistic regression to understand if data has any insight at all.
2. Try getting more data, current dataset is only 11k 
3. Try embedding layer in from of GRU (like some sort of bottleneck)

# Case #2: "bad-or-not-antivirus"
I've tried to guess if there are antiviruses with some anomalies thus we cannot trust them.

To do that I apply t-SNE with `precomputed` distance matrix.
If two antivirus "fire" or "not-fire" together --- distance will be smaller.

![For the first file](https://github.com/tinnulion/bad-domain-names/blob/master/bad_or_not_antivirus/host_detections_tsne_perpl_4.png)

![For the second one](https://github.com/tinnulion/bad-domain-names/blob/master/bad_or_not_antivirus/mal_domains_tsne_perpl_5.png)

Quick takeaway:
There are some antiviruses, which predictions are inconsistent with the majority of others (marked in colors on images).
It might be because they are of bad quality, or just have some bias towards specific malware --- need to dig deeper to understand.

Strange antivirus tools from `host_detections.csv`:
```
"MyWOT": 2,
"SURBL": 2,
"DNS-BH": 3,
"DShield": 3,
"SCUMWARE": 4,
"hpHosts": 4,
"urlQuery": 5,
"GoogleSafeBrowsing": 5,
"DrWeb": 6,
```

Strange antivirus tools from `mal_domains.csv`:
```
"CMC": 6,
"Rising": 6,
"ESET-NOD32": 2,
"NANO-Antivirus": 2,
"TrendMicro": 3,
"TrendMicro-HouseCall": 3,
"ViRobot": 4,
"F-Prot": 4,
"TotalDefense": 4,
"Commtouch": 4,
"VIPRE": 5
```

Besides, there are four large clusters:
1) Avast5 or NOD32
2) MicroWorld-eScan
3) K7AntiVirus
4) MacAfee or Kaspersky

Probably we can use several (two for example) antivirus from a single cluster and got predictions with
same accuracy as with using all antivirus.






