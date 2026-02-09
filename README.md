# GAN-Augmented and LLM-Enhanced Intrusion Detection for Intelligent Vehicles

## Abstract
The rapid growth of connected and autonomous vehicles has greatly expanded the cyberattack surface of invehicle 
networks, especially those based on the Controller Area Network (CAN) bus, which lacks built-in authentication and
encryption. This paper presents a lightweight, semi-supervised intrusion detection system (IDS) that integrates generative
adversarial networks (GANs) and large language models (LLMs) to deliver high accuracy, efficiency, and interpretability. CAN
traffic is transformed into image-like representations that combine identifiers, data length codes, and payload bytes to capture both 
structural and semantic features. A binarized GAN discriminator generates compact, discriminative embeddings that are classified
by a multi-layer perceptron (MLP) under limited supervision. To improve robustness, GAN-based data augmentation enhances
detection stability when trained with as little as 10% of attack data. Experiments on the Car-Hacking and Survival benchmark 
datasets achieve over 99% detection accuracy, outperforming existing baselines while maintaining a total footprint under 10
MB and inference latency below 2 ms. Also, an LLM-driven interpreter produces concise, human-readable explanations for
IDS alerts, increasing operator trust and transparency. The results demonstrate the feasibility of a scalable, explainable, and real-time
IDS for next-generation intelligent vehicles.

## ML Model Architecture
The overall inference incorporates an embedding stage, where the discriminator of a generative adversarial network is 
employed to derive compact yet discriminative representations of CAN traffic.  Classification is then performed using a multi-layer perceptron,
which distinguishes between normal and malicious traffic.
<img width="7420" height="539" alt="image" src="https://github.com/user-attachments/assets/52d5c361-52ee-4a8e-a4c3-d5fd951dae20" />
*Figure 1. General inference architecture.*

## Deployment
In this work, the IDS is deployed on the telematics ECU. If the vehicle integrates a central compute cluster with both
the gateway and telematics, the IDS can be co-hosted there; otherwise, it runs on a stand-alone telematics unit.
<img width="1894" height="849" alt="image" src="https://github.com/user-attachments/assets/7ab7ffb3-2554-4a59-8f21-547cd9f89828" />
*Figure 2. Zonal vehicle architecture with compute cluster.*

## Datasets
Both datasets consist of time-series data recorded from sensors installed in real vehicles.
The data represent CAN bus traffic collected from operational cars. The first dataset was recorded from a Hyundai YF Sonata. 
The second dataset contains measurements collected from three vehicles: Hyundai YF Sonata, Kia Soul, and Chevrolet Spark.

1. [Car-Hacking Dataset, Hacking and Countermeasure Research Lab (HCRL) at Korea University](https://www.kaggle.com/datasets/pranavjha24/car-hacking-dataset)
<img width="770" height="235" alt="car-hacking dataset stats" src="https://github.com/user-attachments/assets/d5e6f079-e0a5-4fb8-b986-71ab1de3b2c7" />
*Table 1. Car-hacking dataset.*
2. [Survival Dataset, Hacking and Countermeasure Research Lab (HCRL) at Korea University]([url](https://ocslab.hksecurity.net/Datasets/survival-ids))
<img width="773" height="217" alt="survival dataset stats" src="https://github.com/user-attachments/assets/14da5de4-f13e-47d9-8fd2-cf361c6c5265" />
*Table 2. Survival dataset.*

## Techniques
1. Baseline represents discriminator embedding + MLP classifier.For classifiers clustering techniques were also used; however, they demonstrate lower generalisability
   across datasets and different vehicles.
3. Augmentations - traditional augmentations, VAE augmentations, GAN augmentations.
   Techniques such as Gaussian noise, upsampling, and sequence reversal improved precision by reducing false positives.
   Variational Autoencoders generated synthetic samples but introduced slight accuracy degradation due to distribution shift,
   with limited variability in reconstructed signals. GANs produced highly expressive and diverse samples and were added for final results.
   <img width="1499" height="380" alt="image" src="https://github.com/user-attachments/assets/1312477c-939f-425c-bc99-a908ab3fb265" />
    *Figure 3. Effect of traditional augmentation on normal CAN traffic.*
4. LLM interpretability feature
   To improve interpretability, an LLM-based post-analysis module was explored.
    Experiments were conducted using GPT-4o, Claude Haiku, DeepSeek, and Mistral-Large. The approach included a prompt constructed from statistical
    characteristics of normal traffic and a comparison with current traffic data, while providing cybersecurity context by framing the model as a
    cybersecurity expert. The prompt was structured according to a persona–task–constraint–context–output format.
   
## Results
### Classification Performance Across Attack Categories  
#### Training MLP on 10% of Attack Training Data (Baseline)
| Metric    | DoS    | Fuzzy  | Gear   | RPM    |
|------------|--------|--------|--------|--------|
| Accuracy   | 99.87% | 99.44% | 99.88% | 99.92% |
| F1 Score   | 99.78% | 99.15% | 99.86% | 99.91% |
| Precision  | 99.98% | 99.91% | 99.96% | 99.99% |
| Recall     | 99.58% | 98.40% | 99.76% | 99.83% |

### Performance on the Survival Dataset  
*(Hyundai Sonata, Kia Soul, Chevrolet Spark)*

#### (a) Accuracy (%)
| Vehicle | Flood | Fuzzy | Malf |
|----------|-------|-------|------|
| Sonata   | 99.95 | 98.88 | 99.57 |
| Soul     | 99.96 | 99.64 | 98.62 |
| Spark    | 99.92 | 97.03 | 99.78 |

---

#### (b) Recall (%)

| Vehicle | Flood | Fuzzy | Malf |
|----------|-------|-------|------|
| Sonata   | 99.90 | 97.32 | 99.07 |
| Soul     | 99.93 | 99.25 | 96.43 |
| Spark    | 99.96 | 90.39 | 99.68 |

### Classification Performance with 10% Training Data and GAN Augmentation

| Metric    | DoS    | Fuzzy  | Gear   | RPM    |
|------------|--------|--------|--------|--------|
| Accuracy   | 99.87% | 99.29% | 99.83% | 99.89% |
| F1 Score   | 99.78% | 98.93% | 99.80% | 99.88% |
| Precision  | 99.99% | 99.99% | 99.99% | 100.00% |
| Recall     | 99.56% | 97.88% | 99.61% | 99.75% |

## Authors
Elizaveta Andrushkevich∗, Zahra Pooranian†, Chuan H. Foh∗, Roc´ıo P´erez de Prado‡, Fabio Martinelli§, Mohammad Shojafar∗

∗ - 6GIC, Institute for Communication Systems, University of Surrey, United Kingdom
† - Department of Computer Science, University of Reading, United Kingdom
‡ - Telecommunication Engineering Department, University of Jaen, Spain
§ - Italian National Research Council (CNR), Pisa, Italy
