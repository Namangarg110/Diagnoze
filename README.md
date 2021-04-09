<p align="center">
<a href="https://github.com/Namangarg110/Diagnoze/blob/master/static/images/logo3.png">
	<img src="https://github.com/Namangarg110/Diagnoze/blob/master/static/images/logo3.png" width=20%/>
</a>
	<h2 align="center"> Diagnoze (Disease Detection)</h2>
	<h4 align="center"> Website which detects different disease</h4>
	<h5 align="center"> TEAM HASH# </h5>
</p>


__Web Site__

<img src="https://github.com/Namangarg110/Diagnoze/blob/master/static/images/screen-capture%20(online-video-cutter.com).gif" />


## Functionalities

- [ ]  Detects types of tumor using MRI SCAN 
- [ ]  Detects Covid-19 using X-RAY
- [ ]  Detects chances of heart attack using general health data 
- [ ]  Detects Diabetes using general health data
- [ ]  User Intractive 
- [ ]  User Friendly


## Tech Stack
* Deep Learning:
	-  Tensorflow
	-  Keras
	 
* Machine Learning:
	-  SK-LEARN

* Data Preprocessing:
	-  Pandas
	-  Numpy
	-  Matplotlib
	-  Pickle
	
* Web:
	-  HTML
	-  CSS
	-  JAVASCRIPT
	-  BOOTSTRAP
	-  FLASK

	

## Introduction
We at Diagnoze aim to provide quick results to the problems faced by our clients.Our webapp works on the principle of taking in info regarding the problem and giving instant results which guide them in taking further step towards treatment.Using this process,we have tried to reduce human errors on any ground.We mainly plan to develop a system where the complexity and computation time is low and accuracy is high.

# Working
*Using Deep Learning and web app we have implimented the followings :- 

## Tumor
* We took Images of MRI Scan from the user as input and then using deep learning,we checked for the presence of tumors in the Image.Then according to the result,we have suggested the user to seek professional medical help.
* Model Used :- Convolutional Neural Network
* Accuracy Achieved :-95.34%
* About Model :- The architecture of the NN consists of 6 Convolutional layers and a couple of Dense Layers.The SGD optimizer was used and the loss function used was binary_crossentropy

## Diabetes
* We have used regular health dataset with features like blood pressure,heart rate,age,gender,chest pain etc. to predict if the patient has diabetes or not.
* Model Used:- Dense Neural Network
* Accuracy Achieved :- 97%
* About Model :- The NN consisted of only 3 Dense Layers. Relu and Sigmoid were used as the activation functions.The Adam optimizer was used and the loss function used was binary_crossentropy

## Heart Attack 
* Using regular health data,we predicted if the patient is at the risk of a heart attack.
* Model Used :- SVM Classifier was used
* Accuracy Achieved :- 87%
* About Model :- Using Grid Search the hyperparameters were tuned. Gamma = 1 , C=10


## Corona Virus 
* Taking in the chest x-ray from the patient and considering the general health details,we tried to anticipate if the user has corona virus.

* **Model Used** :- Fine Tuned MobileNet

* **Accuracy Achieved** :- 99.58%

* **About Model** :- The first 23 layers of the mobilenet architecture were frozen and the rest of the layers were trained using the data,The image was preprocessed according to the mobileNet, The Adam optimizer was used and the loss function used was binary_crossentropy

## Youtube Video
https://www.youtube.com/watch?v=X9wrTBZj0JQ


## Our Team 

<table>
<tr align="center">


<td>

Naman Garg

<p align="center">
<img src = "https://avatars.githubusercontent.com/u/40496687?s=400&u=aeba7754d8bba23a2ab9fb2d794cc316b2b6a84b&v=4"  height="120" alt="Naman Garg">
</p>
<p align="center">
<a href = "https://github.com/Namangarg110"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/naman-garg-3790b917a/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>


<td>

Mudit Jindal 

<p align="center">
<img src = "https://avatars.githubusercontent.com/u/60563356?s=400&u=09a4f1f24803e0bd5cdc674e0fa021ca791fe126&v=4"  height="120"
alt="Mudit Jindal">
</p>
<p align="center">
<a href = "https://github.com/mudit14224"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/mudit-jindal-40521a18b/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>



<td>

Tanishq Kumar

<p align="center">
<img src = "https://avatars.githubusercontent.com/u/66270248?s=400&u=970a9ef7dcdc609ab393c89d5bef50fb63380af5&v=4"  height="120" alt="Breenda Das">
</p>
<p align="center">
<a href = "https://github.com/tanishq20"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/tanishq-kumar-b03a52194/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>

<td>

Amisha Jaiswal

<p align="center">
<img src = "https://avatars.githubusercontent.com/u/66247959?s=400&u=9d53158da177d70996607715a9fb2cd2e9ad8214&v=4"  height="120"
alt="Amisha Jaiswal">
</p>
<p align="center">
<a href = "https://github.com/amishajais21"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/amisha-jaiswal-8532b1169/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>
</tr>

<td>

Dhanya Sri Aravapalli

<p align="center">
<img src = "https://avatars.githubusercontent.com/u/71751040?s=400&u=18a3a39e283646ff410a2032c216cc97ec0529ca&v=4"  height="120"
alt="Dhanya Sri Aravapalli">
</p>
<p align="center">
<a href = "https://github.com/Dhanya-26"><img src = "http://www.iconninja.com/files/241/825/211/round-collaboration-social-github-code-circle-network-icon.svg" width="36" height = "36"/></a>
<a href = "https://www.linkedin.com/in/dhanya-sri-aravapalli-70a6851a5/">
<img src = "http://www.iconninja.com/files/863/607/751/network-linkedin-social-connection-circular-circle-media-icon.svg" width="36" height="36"/>
</a>
</p>
</td>
</tr>
  </table>

<p align="center">
	Made with :heart: by Team HASH</a>
</p>
