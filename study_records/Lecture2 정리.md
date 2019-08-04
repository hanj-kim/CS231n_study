Lecture 2. Image Classification
===============================

Fei-Fei 교수님의 cs231n (Spring 2017) 강의를 정리해보고자 합니다. 

(본 포스팅은 cs231n 강의 Slide를 참고하여 작성하였습니다.)

강의 자료는 아래 링크를 참고하면 됩니다.

Youtube: [https://youtu.be/vT1JzLTH4G4](https://youtu.be/vT1JzLTH4G4)

Course Notes: [http://cs231n.github.io/](http://cs231n.github.io/)

#
>Lecture 2. formalizes the problem of image classification. We discuss the inherent difficulties of image classification, and introduce data-driven approaches. We discuss two simple data-driven image classification algorithms: K-Nearest Neighbors and Linear Classifiers, and introduce the concepts of hyperparameters and cross-validation.

#
![1](https://user-images.githubusercontent.com/37207332/61455680-34120080-a99f-11e9-9348-cda6b8a2c4b9.JPG)


Computer Vision에서는 image를 어떻게 Classification(분류)할까?라는 문제인데 사람의 경우 사진을 보고 바로 '고양이'라고 대답을 할 것입니다. 하지만 컴퓨터는 아래와 같이 image를 픽셀 정보로 보게 됩니다. 픽셀은 height(255) x width(255) x channel(3)으로 이뤄져있고, 컴퓨터는 각 픽셀에 대한 숫자 값으로 image를 보게 되는 것입니다.
#
![2](https://user-images.githubusercontent.com/37207332/61455721-53a92900-a99f-11e9-8c9e-0276399ec568.JPG)


그렇기 때문에 몇 가지 Challenges(문제)가 발생하는데, 하나씩 살펴보겠습니다.
#
![3](https://user-images.githubusercontent.com/37207332/61455739-602d8180-a99f-11e9-86fb-ea1d8443c578.JPG)


우선 보는 관점, 위치에 따라서 image가 달라질 수 있다는 문제가 있습니다.
#
![4](https://user-images.githubusercontent.com/37207332/61455768-776c6f00-a99f-11e9-84e1-ca3529c74d4a.JPG)

다음으로는 illumination은 빛이 비치는 양에 따라서 달라보일 수 있다는 문제가 있습니다.
#
![5](https://user-images.githubusercontent.com/37207332/61455842-a1be2c80-a99f-11e9-9653-bc6dbdcbd57b.JPG)

같은 고양이라도 그 모습의 변형으로 인해 다르게 보일 수 있다는 문제가 있습니다.

사람은 4개 모두 같이 고양이로 보지만 컴퓨터는 다르다고 판단할 수도 있는 것입니다.
#

![6](https://user-images.githubusercontent.com/37207332/61455843-a1be2c80-a99f-11e9-91f9-4a62149b7be5.JPG)

Occlusion은 형체의 일부만 보이는 것을 말합니다.

위 그림의 세번 째 그림처럼 꼬리만 보이는 경우는 고양이라고 거의 판별할 수 없을 것입니다.
#

![7](https://user-images.githubusercontent.com/37207332/61455844-a256c300-a99f-11e9-92aa-b5a72631267c.JPG)

Background Clutter는 객체가 배경색과 거의 비슷하여 배경과 구분을 할 수 없는 문제입니다.
#
![8](https://user-images.githubusercontent.com/37207332/61455845-a256c300-a99f-11e9-8d9b-49efd0114542.JPG)

Intraclass variation은 intra는 안의, 내부의 라는 뜻으로 보면되고, 같은 클래스 내에서도 다른 색, 형태가 있을 수 있다는 것입니다.

Deformation과의 차이점은 Deformation은 포즈가 다릅니다. 즉 모습의 형태가 다른 것(변형)을 말하는데

intraclass variation은 위의 그림처럼 같은 새끼고양이라도 색이 다르거나 줄무늬가 다르거나 하는 것을 말합니다.
#
![9](https://user-images.githubusercontent.com/37207332/61455847-a256c300-a99f-11e9-8927-1c52ae791403.JPG)

#

![10](https://user-images.githubusercontent.com/37207332/61455848-a2ef5980-a99f-11e9-8bb2-db460542e6e4.JPG)

여기서는 image에 label을 붙여서 하는 방식인 Data-Driven Approach(데이터에 기반한 접근)방법을 소개하고 있습니다.

1\. image에 label을 붙인 Dataset을 모은다.

2\. 머신러닝을 이용하여 분류하는 것을 학습한다.

3\. 새로운 image를 학습시킨 모델에 넣어 성능을 테스트한다.
#
![11](https://user-images.githubusercontent.com/37207332/61455849-a2ef5980-a99f-11e9-8755-632ab3123d12.JPG)

첫번째 Image Classification 방법은 Nearest Neighbor입니다. 방법은 간단합니다.

image에 label을 붙여서 모든 Image를 저장하고, 저장되지 않은 새로운 image를 모델에 넣으면

모델은 저장된 image 전부와 새로운 image를 하나씩 비교하면서 가장 비슷한 image를 찾습니다.
#
![12](https://user-images.githubusercontent.com/37207332/61455850-a2ef5980-a99f-11e9-8510-3fef4e5071e3.JPG)

강의에서는 CIFAR10이라는 데이터셋을 사용하여 예를 들어주었습니다.

CIFAR10은 10개의 클래스로 되어있고, 5만장을 학습시킨 것이고, 1만 장의 image는 test를 하였습니다.

오른쪽을 보게 되면 Image Classification 성능은 좋지 않다는 것을 알 수 있습니다.

맨 위는 비행기를 넣었는데 말, 배 그리고 하얀색 물체들이 나왔습니다.

개구리를 넣었을 때는 개구리와 다른 전투기, 고양이 등이 나오는 것을 볼 수 있습니다.
#
![13](https://user-images.githubusercontent.com/37207332/61455851-a387f000-a99f-11e9-91f9-9815f876391e.JPG)

Nearest Neighbor 방식은 간단하게 각 픽셀 간의 차이(거리)로 image를 분류합니다.

여기서는 각 픽셀 값들의 차의 절댓값으로 표현하는데 이유는 test image가 저장된 image와 얼마만큼의 차이가 있는지만 비교하면 되기 때문에 -값은 필요가 없이, 모든 거리의 차이 값만 더해서 확인하면 됩니다.
#
![14](https://user-images.githubusercontent.com/37207332/61455852-a387f000-a99f-11e9-93a9-2138aec927a8.JPG)

여기서 Nearest Neighbor 사용할 때 N개의 예제 image가 있다면 training과 prediction은 얼마나 빠를까를 물어보는데, 

Nearest Neighbor는 training은 image를 저장만 하기 때문에 복잡도가 O(1)이고,

prediction은 모든 저장된 image와 비교하기 때문에 복잡도가 O(N) (Number of stored Image)으로 오래 걸린다는 것입니다.

우리가 원하는 건 새로운 image를 넣었을 때 그것이 어떤 image인지 빨리 찾아주기를 원하는데

Nearest Neighbor의 경우 학습은 매우 빠르나(라벨만 붙여서 저장만 하면 되기 때문),

새로운 image가 어떤 것인지 찾는 것은 저장된 image들과 하나하나 모두 다 비교해야하기 때문에 매우 느립니다.
#
![15](https://user-images.githubusercontent.com/37207332/61455853-a4208680-a99f-11e9-8955-71166169ee8b.JPG)

이어서 K-Nearest neighbors 개념이 나옵니다. KNN이라고 합니다.

위의 그림에서 점들은 image를 의미하고, 그 image와 가까운 거리(비슷한 image)들과 그룹을 짓게 됩니다.

그룹을 만드는 방법은 간단하게 K의 값은 자신의 image(새로운 image)로부터 가까운 image 갯수를 의미한다.

예를 들어, 새로운 image가 초록색과 노란색 경계선 부분에 들어갔다고 생각해보자.

그 때 k=3이라고 가정했을 때, 초록색 2 / 노란색 1 이렇게 선택이 되었다면, 과반수로 인해서 새로운 image는 초록색 분류의 image가 된다.

맨 왼쪽의 K=1인 경우는 경계선도 매끄럽지 못하고, 경계선이 서로 맞닿고 구분이 쉽지 않을 것이다.

그리고 초록색 부분 가운데에 뜬금없이 노란색 점이 하나 있는데 이것을 outlier(이상치, 오류값)이라고 한다.

K값을 어떻게 설정하느냐에 따라 classifier의 정확도가 결정된다.
#
![16](https://user-images.githubusercontent.com/37207332/61455854-a4208680-a99f-11e9-8c3e-4a97690278cf.JPG)

KNN의 Distance를 결정하는 방법은 두 가지가 있을 수 있는데 L1과 L2이다.

먼저 L1 distance 방식이 있는데 Manhattan distance라고도 하며, 각 좌표 값들의 거리 합이다.

L2는 Euclidean distance라고 하며 제곱의 합의 루트이기 때문에 좌표 상에서 두 점간의 최단 거리를 의미한다.

그래서 L2 distance가 원형 모양으로 좀 더 매끄럽게 나오는 것을 알 수 있다.

L1을 사용할 것인지 L2를 사용할 것인지는 일종의 hyperparameter라고 합니다.
#
![17](https://user-images.githubusercontent.com/37207332/61455857-a4208680-a99f-11e9-9dd5-8bb192cb898e.JPG)

L1은 경계선 부분이 울퉁불퉁하고, 거칠게 보이는데 L2는 매끄럽고 경계선 윤곽이 뚜렷하다.
#
![18](https://user-images.githubusercontent.com/37207332/61455860-a4b91d00-a99f-11e9-8781-462ace4a2f4a.JPG)

그렇다면 K값과 distance 값은 어떻게 정해줘야 가장 좋을까?라는 질문을 합니다.

그 답은 두 값 모두 우리가 정해가면서 최적의 값을 찾아야한다고 합니다. 그래서 최적의 값인 hyperparameter를 찾는 것이 목표라고 합니다.
#
![19](https://user-images.githubusercontent.com/37207332/61455862-a4b91d00-a99f-11e9-87eb-8614801da148.JPG)

Hyperparameter를 찾는 방법을 4가지 정도 소개하는데, Idea 1은 보면 Dataset만 있는 경우입니다.

K=1인 경우로, train set과 test set이 동일하여 항상 정확도는 100이지만 만약 새로운 image가 들어올 경우에는 정확도는 매우 낮을 것입니다.

Idea 2는 train set과 test set으로 나누는 것인데요. 여기에도 문제점이 있습니다.

열심히 parameter를 정해서 train을 하고 test를 할 때 우리가 학습한 model이 정확도가 얼마나 나올지 모릅니다.

test는 실제로 우리가 적용할 data를 의미하기 때문에 우리의 학습 모델이 실제 image에 얼마만큼의 정확도가 나오는지 알고 적용을 하는 것이 중요합니다.

그래서 나온 것이 Idea3입니다.

전체 데이터 중 train set에서 학습을 시키고, 학습된 model을 validation set을 통해 hyperparameter tuning을 한 후 최종 model을 test set에 적용하여 성능을 확인하는 것입니다.
#
![20](https://user-images.githubusercontent.com/37207332/61455863-a4b91d00-a99f-11e9-83ca-d09edbc4df19.JPG)

마지막으로 Idea 4인 Cross-Validation을 설명하는데요. 이 방법은 Data set의 크기가 작을 때는 효과적이지만,

Data set의 크기가 커지면 비효율적이라고 합니다. 이유를 살펴보면

Cross-validation은 fold 개념을 추가하여 전체 training set을 N 등분하여 fold(1~n)를 만듭니다.

그리고 각 fold 부분이 돌아가면서 (N-1)개의 train set, 1개의 validation set이 되어 train을 하고 validation을 하게 됩니다. 그렇기 때문에 data set의 크기가 커지면 training과 prediction이 매우 오래 걸릴 수 있다는 문제점이 있습니다.
#
![21](https://user-images.githubusercontent.com/37207332/61455864-a551b380-a99f-11e9-9a0b-10e399f2c976.JPG)

하지만 KNN은 Image Classification에서는 절대 사용하지 않는다고 말합니다. 
이유는 2가지입니다.
첫 번째는 test time 즉, prediction 시간이 오래걸린다는 것입니다.

두 번째는 위의 사진들을 보면 원본, 일부를 가린 것, Shifted, Tinted처럼 4가지는 모두 다른 사진입니다.하지만 KNN은 이들을 모두 같은 사진으로 본다는 것입니다. 그래서 image Classification에서는 사용하지 않는다고 합니다. 

#
지금까지 내용을 정리하자면,

1\. computer는 image를 pixel(digit)로 본다.

그래서 몇 가지 challenge가 있다.(Viewpoint variation, lllmination, Deformation, Occlusion, Background Clutter, IntraClass variation)

2\. Image Classification은 image에 label을 붙여 저장한 training set으로 모델을 학습시키고, test set으로 모델이 label을 예측합니다.

3\. prediction은 train image와 test image의 pixel 값들의 차이(distance)로 한다.

4\. 3번 방법은 Nearest Neighbor방법이라고 하고, KNN방법이 있다.

5\. KNN에서 K는 hyperparameter이다.

6\. 하지만 image Classification에서는 KNN을 절대 사용하지 않는다.
#
다음은 Linear Classification 내용을 살펴보겠습니다.

![2](https://user-images.githubusercontent.com/37207332/61456057-1d1fde00-a9a0-11e9-9dd4-ef0cb71e1cce.JPG)

Image Classification은 왼쪽 그림과 같이 어떤 image에 대한 정보를 text로 보여주는 것(image Captioning)도 가능할 것입니다. image captioning은 CNN + RNN 형태이고, CNN은 classifier / RNN은 문장구성을 하여 image captioning을 한다고 합니다. 또는 image를 다음 아래 그림과 같이 나누는 것(image segmentation)도 가능할 겁니다.

![3](https://user-images.githubusercontent.com/37207332/61457154-a6381480-a9a2-11e9-8e7b-04d7375a4cdd.JPG)

또는 object Detection도 가능합니다.

![4](https://user-images.githubusercontent.com/37207332/61457155-a6381480-a9a2-11e9-99b0-c12cd13aa779.JPG)
#
cs231n을 통해 Object Detection, Image Segmentation, Image Captioning 이런 것들을 어떻게 computer가 수행하는지에 대해서 배우게 됩니다.
#
![5](https://user-images.githubusercontent.com/37207332/61457156-a6d0ab00-a9a2-11e9-8996-c52e734d387b.JPG)

이번에도 CIFAR10 Dataset을 이용합니다.
#
![6](https://user-images.githubusercontent.com/37207332/61457157-a6d0ab00-a9a2-11e9-9905-744018f45c4e.JPG)

지난 시간에 배운 Data-Driven Approach 방식과 달리 이번 시간에는 Parametric Approach 방법을 소개하고 있습니다.

이번 방법은 f(x,W) = Wx라는 선형식을 이용하여 image 각 픽셀 값 32x32x3을 일렬로 쭉 뽑아서 W(weight) (각 픽셀 값들의 가중치)를 부여하여 클래스를 정하는 방법입니다.
#
![7](https://user-images.githubusercontent.com/37207332/61457158-a6d0ab00-a9a2-11e9-9f18-8cfb46fb0049.JPG)

위 수식에 b(bias)를 추가하여 수식을 세울수도 있습니다.
#
![8](https://user-images.githubusercontent.com/37207332/61457159-a6d0ab00-a9a2-11e9-9e3b-eafd5bc406fc.JPG)

위 수식이 어떻게 동작하는지 살펴보면, 고양이의 image 픽셀이 4개만 있다고 가정한 것입니다.

4개가 있을 때 이를 일렬로 쭉 뽑아서 하나의 column으로 우측에 놓고, 좌측의 각 가중치들과 곱하여 bias를 더한 결과가 우측항입니다. 즉, image에 각 픽셀 값들에 w 곱한 행렬 계산을 한 결과가 우측 항이 됩니다. 

row별로 색이 다른 것을 확인 할 수 있는데 이는 클래스를 의미합니다.

맨 위는 cat class, 가운데는 dog class, 맨 밑은 ship class입니다.

위 그림은 고양이 그림을 넣었는데 Dog의 점수가 가장 높으므로 제대로 classification을 하지 못했다고 할 수 있습니다.

이 때는 w값을 조정해서 해당 image를 정확히 찾도록 해주어야하는데 이 부분은 앞으로 강의가 진행하면서 나오게 됩니다.
#
![9](https://user-images.githubusercontent.com/37207332/61457160-a7694180-a9a2-11e9-9415-51dbc78063ef.JPG)

위 그림은 CIFAR10을 Wx + b로 train 했을 때의 W 값들을 아래 흐릿한 사진들로 보여주고 있습니다.

말의 경우 머리가 양쪽으로 있는데 아마도 사진을 학습시킬 때 방향이 좌,우 모두 있었던 것 같습니다.

car의 경우 빨간색 차가 많았던 것 같네요. 학습 image가 어떠냐에 따라서 train 결과가 달라지겠습니다.
#
![10](https://user-images.githubusercontent.com/37207332/61457161-a7694180-a9a2-11e9-9012-acde208c8489.JPG)

Linear Classifier는 위 그림에서 보시는 것처럼 선을 그어 특정 image를 분류하는 것으로 볼 수 있습니다.
#
![11](https://user-images.githubusercontent.com/37207332/61457162-a7694180-a9a2-11e9-9bc7-3fef44a480b9.JPG)

하지만 단순히 wx+b라는 직선으로는 위 그림과 같은 문제는 해결할 수 없습니다.

어떤 데이터가 대각선으로 있는 경우는 하나의 선으로는 나눌 수가 없을 것입니다.

원형의 형태도 하나의 직선으로 나눌 수 가 없을 것입니다.

세 번째 그림처럼 특정 부분에만 있는 경우에도 직선으로는 나눌 수 없을 것입니다.

Linear Classification으로는 완벽하게 image를 분류할 수 없다라고 생각할 수 있습니다.

이를 해결하는 문제는 앞으로 Lecture가 진행되면서 배우게 됩니다.
#
![12](https://user-images.githubusercontent.com/37207332/61457163-a7694180-a9a2-11e9-92cd-9ab37e0e2d47.JPG)

위 그림은 Linear Classification으로 cat, car, frog를 test한 것입니다.

여기서 어떤 것이 가장 좋고, 나쁘냐?라고 묻고 있는데, 먼저 cat을 보면 2.9 점수가 나왔습니다. 하지만 두 칸 밑에 dog가 8.02로 가장 높은 것을 확인할 수 있어서 제대로 분류하지 못했다고 할 수 있습니다.

car의 경우를 보면 6.04로 가장 높은 점수가 나왔습니다. 즉 제대로 분류했다고 볼 수 있겠습니다.

frog의 경우를 보면 -4.34로 가장 낮은 점수인 -4.79 다음의 점수가 나왔기 때문에 가장 분류를 못했다고 볼 수 있습니다.

정리하자면, 이번 시간에는 Linear Classification에 대해서 배웠습니다.

Linear Classification은 image의 각 픽셀 값에 w(weight)를 곱한 값들의 합 즉, 행렬 연산을 통해 score를 얻게 되는 방식입니다. W값을 조정하면서 Classification 성능을 높일 수 있습니다. 하지만 직선이기 때문에 직선으로 나눌 수 없는 Non-Linear한 경우들에서는 image Classification을 해결하지 못합니다.
