# CSI-Cloak

---

### Description

Passive device-free localization of a person exploiting the Channel State Information (CSI) from Wi-Fi signals is quickly becoming a reality. While this capability would enable new applications and services, it also raises concerns about citizens’ privacy. In this work, we propose a carefully-crafted obfuscating technique against one of such CSI-based localization methods. In particular, we modify the transmitted I/Q samples by leveraging an irreversible randomized sequence. I/Q symbol manipulation at the transmitter distorts the location-specific information in the CSI while preserving communication, so that an attacker can no longer derive information on user’s location. We test this technique against a Neural Network (NN)-based localization system and show that the randomization of the CSI makes undesired localization practically unfeasible. Both the localization system and the randomization CSI management are implemented in real devices. The experimental results obtained in our laboratory show that the considered localization method (first proposed in an MSc thesis) works smoothly regardless of the environment, and that adding random information to the CSI mess up the localization, thus providing the community with a system that preserve location privacy and communication performance at the same time.

---

### Repo Capabilities


- ... 

- [Device Free Indoor Localization](https://github.com/seemoo-lab/csicloak/tree/master/Device_Free_Indoor_Localization):

	- Train a Neural Network to localize a human inside a room by only exploiting the CSI information of the transmitted packets

	- Capabilities:
	
		- The import of raw CSI information and the conversion in different formats: Amplitude-phase, amplitude-unwrapped phase and IQ data.
	
		- The baselining of CSI information against a measurement over cables or the measured empty room. This improves the CSI quality and removes the noise introduced by the hardware
		
		- A variety of preprocessing options, including the selection of the number of subcarriers, TX/RX antennas, removal of special subcarriers like the pilot subcarriers and normalization
		
		- The definition of rooms offers a simple way to describe different environments. These can be further constrained by using masks.
	
		- The user can create a Neural Network and define own loss functions to approach each problem individually. The usage of [Tune](https://docs.ray.io/en/latest/tune.html) offers a scalable and fast hyperparameter Tuning.
		
		- A trained Neural Network can be analyzed in various ways. A CDF shows the detailed performance of the classifier. Moreover, a heatmap shows the performance dependent on the true location and a hitmap visualized the estimation areas. Furthermore, the best and worst estimated can be identified and visualized on the room.
		
		- Besides the classical end-to-end training, an existing network can also be re-trained based on new measurements to adapt to a new room.
		
		- A real-time mode allows to observe the performance of the classifier in real-time
		
		- The storage of the used random permutations assures reproducability
		
		- Finally, the framework offers detailled human-readable logs and an own crawler to extract the required information fast and well-arranged.
		- Detailled information about the project and capabilities can be found in the  Master Thesis [Device-Free Indoor Localization: A User-Privacy Perspective](https://www.researchgate.net/publication/343222355_Device-Free_Indoor_Localization_A_User-Privacy_Perspective?channel=doi&linkId=5f1d84c645851515ef4afdf4&showFulltext=true&fbclid=IwAR2GANVyN2P-IFOCS59a8w-ceVmxSb48zFx-lH24sGK75pyeRvtk2uxwCyw)
---
			
### Usage:

- ...

- [Device Free Indoor Localization](https://github.com/seemoo-lab/csicloak/tree/master/Device_Free_Indoor_Localization):

	- Three python files are callable:
	
		- *main.py*: contains most of the described features.
		
		- *optimizationTune.py*: offers a fast and scalable training by using [Tune](https://docs.ray.io/en/latest/tune.html) 
		
		- *LogAnalyzer.py*: allows you to quickly search different type of logs
		
	- All of them are called without any parameters. The parameters are set in the **parameters.py* files. 
	
		
			python main.py
		
	
		- Detailled information about the different modes and all parameters can be found in the Appendix of the Master Thesis [Device-Free Indoor Localization: A User-Privacy Perspective](https://www.researchgate.net/publication/343222355_Device-Free_Indoor_Localization_A_User-Privacy_Perspective?channel=doi&linkId=5f1d84c645851515ef4afdf4&showFulltext=true&fbclid=IwAR2GANVyN2P-IFOCS59a8w-ceVmxSb48zFx-lH24sGK75pyeRvtk2uxwCyw)

### References

---

- This repository contains the source code of [An Experimental Study of CSI Management to Preserve Location Privacy](TODO) by M. Cominelli, F. Kosterhon, F. Gringoli, R. Cigno and A. Asadi

- The source code is based on the Master Thesis [Device-Free Indoor Localization: A User-Privacy Perspective](https://www.researchgate.net/publication/343222355_Device-Free_Indoor_Localization_A_User-Privacy_Perspective?channel=doi&linkId=5f1d84c645851515ef4afdf4&showFulltext=true&fbclid=IwAR2GANVyN2P-IFOCS59a8w-ceVmxSb48zFx-lH24sGK75pyeRvtk2uxwCyw) by F. Kosterhon

- The CSI-Extraction uses the [Nexmon-CSI](https://github.com/seemoo-lab/nexmon_csi) project developed by F. Gringoli, M. Schulz and J. Link.

---

### Contact

---

...

### Powered By

---

#### Secure Mobile Networking Lab (SEEMOO)

...

#### Multi-Mechanisms Adaptation for the Future Internet (MAKI)

...

#### LOEWE centre emergenCITY


...

#### Technische Universität Darmstadt

...

#### University of Brescia

...