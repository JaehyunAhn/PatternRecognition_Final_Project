## 패턴인식 기말 프로젝트

---
### 프로젝트 기본정보

- 지은이:	Jaehyun Ahn (jaehyunahn@sogang.ac.kr)
- 프로젝트: 패턴인식(Pattern Recognition) 기말 프로젝트
- 프로젝트 제출일: 15/06/19


### 버전 정보
	- Python 2.7.6
	- Numpy 1.9.2
	- Scikit-learn 0.16.1
	- Pandas 0.16.1
	- Nolearn 0.5
	
### 입력 데이터

- 학습데이터: 100명의 서로 다른 사용자, 정면 무표정 데이터(100장, 각 512x768 해상도)

- 테스트데이터: 100명 서로 다른 사용자(테스트 데이터에 있는 사람들과 동일)의 정면 웃는 사진, 우측면 25도 무표정 사진, 좌측면 25도 무표정 사진(각 100장)

- 추가 데이터: 100명의 서로 다른 사용자들의 눈 좌표값(수작업 입력)


### 출력 데이터

 각 100명의 이미지를 분류하고 테스트 데이터로부터 출력 결과값을 저장받고 그 정확도를 측정하고 분석
 
---

### 파일 설명
 
- **CNNModule.py / CNNmodel.py** , Convolutional Neural Networks 테스트 데이터**(실제 동장하지 않음)**
- **DBNModule.py / DBNModel.py** , Deep Belief Networks 코드.
- **MNIST_handwritten_Recog.py** , MNIST 손글씨 인식 테스트를 개조하여 본 실험을 수행. Random Forest, Linear SVM, Nearest Neighbor Classification 기법을 사용
- **README.md** , 프로젝트 설명이 기술된 파일
- **FERET_subset**, 테스트 데이터가 입력되어 있는 파일. 파일 형식은 jpg로, *fa* 폴더에는 정면 사진이, *fb* 폴더에는 정면 웃는 사진이, *ql* 폴더에는 좌측 무표정 25도 측면 사진이, *qr* 폴더에는 우측 무표정 25도 측면 사진이 저장되어 있다

### 프로젝트 세팅 및 수행결과

100개의 데이터로 학습 모델을 만들고 테스트 데이터 세트(정면 웃는 얼굴, 우측 25도, 좌측 25도 사진 각 100장씩 총 300장)로 학습을 한 결과는 다음과 같다.

#### MNIST_handwritten_Recog.py

##### Random Forest 
	Random Forest Accuracy:  0.156666666667
	RF Precision: 47 / 300 ( 15.6666666667 % ) 

##### Linear Support Vector Machine
	Linear SVM Accuracy:  0.48
	SVM Precision: 144 / 300 ( 48.0 % ) 

##### Nearest Neighbor Classification
	Nearest Neighbor Accuracy:  0.106666666667
	KNN Precision: 32 / 300 ( 10.6666666667 % ) 
	
##### Deep Belief Networks
	DBN(Deep Belief Networks) Precision: 129 / 300 ( 43.0 % )

  