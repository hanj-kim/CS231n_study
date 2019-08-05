Lecture3. Loss Functions and Optimization
=========================================

Fei-Fei 교수님의 cs231n(Spring 2017) 강의를 정리해보고자 합니다.

(본 포스팅은 cs231n 강의 Slide를 참고하여 작성하였습니다.)

강의 자료는 아래 링크를 참고하면 됩니다.

Youtube: [https://youtu.be/vT1JzLTH4G4](https://youtu.be/vT1JzLTH4G4)

Course Notes: [http://cs231n.github.io/](http://cs231n.github.io/)

#
>Lecture 3 continues our discussion of linear classifiers. We introduce the idea of a loss function to quantify our unhappiness with a model’s predictions, and discuss two commonly used loss functions for image classification: the multiclass SVM loss and the multinomial logistic regression loss. We introduce the idea of regularization as a mechanism to fight overfitting, with weight decay as a concrete example. We introduce the idea of optimization and the stochastic gradient descent algorithm. We also briefly discuss the use of feature representations in computer vision.
#

지난 시간에 computer vision에서 이미지를 분류할 때의 Challenge와 컴퓨터에게 어떻게 image를 학습시킬 것인지에

대해서 배웠습니다. 그 방법으로는 data-driven approach 방법과 parametric approach 방법이 있었습니다.

그리고 마지막에 Linear Classifier에 대해서 내용이 나오면서 f(x) = wx + b를 통해서 하나의 직선을 그어서

이미지를 분류하는 것까지 다뤘습니다.

![1](https://user-images.githubusercontent.com/37207332/61459034-d8e40c00-a9a6-11e9-8614-b5e4fb369812.JPG)
#

이번 강의에서는 Loss function과 Optimization에 대해서 배웁니다.

loss function은 위 그림처럼 각 클래스에 대한 점수가 나왔을 때 그 점수가 안좋다면 점수가 좋지 않다고 말해줍니다.

따라서 loss function의 결과 값에 따라서 원하는 결과가 나올 수 있도록 parameter들을 조정할 수 있게 되는데

이것을 optimization이라고 보면 되겠습니다.

loss function을 통해 클래스 분류가 잘 이뤄지고 있는지를 판단하고, 잘 되고 있지 않다면 optimization을 통해서

분류가 잘 이뤄질 수 있도록 W값을 조정하는 것입니다.

![2](https://user-images.githubusercontent.com/37207332/61459036-d97ca280-a9a6-11e9-8a6b-80a618bf1395.JPG)
#

Loss function은 L로 표시되는 식으로 나타낼 수 있고, Li(x) 함수에 대한 출력 값들에 합의 평균으로 나타낼 수 있습니다.

![3](https://user-images.githubusercontent.com/37207332/61459037-d97ca280-a9a6-11e9-9c4d-40f8dbb97d3e.JPG)
#

첫번 째 Li 함수로 Multiclass에서 사용되는 SVM(Support Vector Machine)을 설명합니다.

S<sub>j</sub>는 정답이 아닌 클래스이고, S<sub>y<sub>i</sub></sub>는 정답인 클래스 입니다.

위 그림에서 cat을 살펴보면 S<sub>y<sub>i</sub></sub>는 3.2가 되고, S<sub>j</sub>값은 car: 5.1, frog: -1.7이 될 것입니다.

![4](https://user-images.githubusercontent.com/37207332/61459038-d97ca280-a9a6-11e9-8661-92443b8b0955.JPG)
#

SVM은 그래프 모양이 Hinge(경첩)과 비슷하다고 하여 Hinge Loss라고도 합니다.

S<sub>y<sub>i</sub></sub>값이 크면(정답 클래스 점수가 높게 나온다면) S<sub>j</sub> - S<sub>y<sub>i</sub></sub> +1 값이 1(Safety Margin)보다 작아질 것이고,

따라서 그래프처럼 Loss가 0이 됩니다. 즉, '정답 클래스 값이 정답이 아닌 클래스, Safety Margin보다 크다면

Loss는 0이다.'라고 보는 겁니다.

![5](https://user-images.githubusercontent.com/37207332/61459040-d97ca280-a9a6-11e9-8394-7f7a81c94caa.JPG)
#

강의 내용을 보면,

여기서 Safety Margin을 두는 이유가 나오는데, Safety Margin이 무엇이냐면 만약에 S<sub>j</sub>와 S<sub>y<sub>i</sub></sub>값이 같은 값이 나오면

Loss가 있음에도 불구하고, 그 값이 max(0,0)이 나오게되는 상황이 나오기 때문에 Safety Margin을 두게 됩니다.

예를 들면 S<sub>j</sub> = 3.2, S<sub>y<sub>i</sub></sub> = 3.2이면 3.2 - 3.2 = 0이 나와서 max(0,0)이 되는 것이죠.

클래스 분류가 제대로 되지 않았음에도 loss가 0이 나올 수 있기 때문에 이를 방지하기 위한 값이라고 보시면 됩니다.

그리고 Safety Margin 값은 hyperparameter로 원하는 값을 조정해서 넣을 수 있는데,

만약 margin값이 크다면 S<sub>y<sub>i</sub></sub>(정답 클래스)값이 더 커져야만 loss가 0이 될 수 있을 것이고,

margin 값이 작다면 S<sub>y<sub>i</sub></sub>(정답 클래스) 값이 더 작아도 loss가 0이 될 수 있게 됩니다.
#

다음은 강의에서 나오는 몇 가지 질문들을 살펴보겠습니다.

![6](https://user-images.githubusercontent.com/37207332/61459042-da153900-a9a6-11e9-97a8-ce03dab85044.JPG)

만약에 S<sub>j</sub> - S<sub>y<sub>i</sub></sub>에서 j=y\_i인 값들도 모두 포함해서 계산하면 어떻게 되는가? 라는 질문인데

이는 자신과 다른 클래스의 차이를 구하는 것뿐만 아니라 자신과 자신의 차이도 구하면 어떻게 되는가?라는 질문입니다.

자신과 자신의 차이는 항상 0이 되므로 Safety Margin 값이 한번 더 추가되서 나오게 됩니다.
#
![7](https://user-images.githubusercontent.com/37207332/61459043-da153900-a9a6-11e9-80bc-11f3b8a4546c.JPG)

만약에 sum(합)을 하지 않고, max 값들의 평균으로 하면 어떨까? 라는 질문에는 상관없다.

우리가 구하자고 하는 것은 어차피 현재 정답 클래스가 잘 분류되고 있는지에 대한 수치를 알고 싶은 것이기 때문에

수치에 대한 크기는 상관이 없습니다.
#
![8](https://user-images.githubusercontent.com/37207332/61459044-da153900-a9a6-11e9-9186-266e9f9f757a.JPG)

다음 질문은 max 값에 제곱을 하면 어떻겠느냐? 라는 질문인데, 이 질문에 답은 '안된다'입니다.

왜냐하면 제곱을 하게 되면 그래프가 Nonlinear하게 되므로 Linear하지 않게 된다는 것입니다.
#
![9](https://user-images.githubusercontent.com/37207332/61459045-daadcf80-a9a6-11e9-9fa2-827ea94f451a.JPG)

다음은 우리가 찾고자하는 Loss가 0이되는 W값은 유일할까?라는 질문에

'아니다 W를 2배를 해도 Loss는 0이 나온다.'라는 것입니다.

예를 보시겠습니다.

![10](https://user-images.githubusercontent.com/37207332/61459047-daadcf80-a9a6-11e9-85e8-93df53610fdb.JPG)

위와 같이 W 값을 2배를 해도 Loss는 0으로 같은 값이 나오는 것을 확인할 수 있습니다.

우리가 여기서 해야하는 것은 Loss를 0으로 만드는 유일한 W을 구하는 것입니다.

따라서 Regularization을 통해서 w값이 unique하게 나오도록 해야합니다.
#
![11](https://user-images.githubusercontent.com/37207332/61459048-daadcf80-a9a6-11e9-8aea-d70ac1784312.JPG)

위 그림을 보시면 파란색 선들이 있는데 이게 현재 학습된 W값이고, 초록색은 새로운 데이터라고 보시면 됩니다.

Regularization이 되어있지 않으면 기존의 W 값이 unique하지 않아서 비선형적으로 되고, 따라서 새로운 데이터가

들어왔을 때 그 값을 쉽게 유추하지 못하게 된다는 뜻입니다.

위 그림처럼 Lambda 값을 추가해서 기존 모델을 비선형에서 더 단순하게 만들어야하고,

차원(Dimension)을 줄인다고 볼 수 있겠습니다.
#
![12](https://user-images.githubusercontent.com/37207332/61459049-db466600-a9a6-11e9-97e8-1a173d2b4c21.JPG)

Regularization에 쓰이는 함수들을 설명하고 있는데 보통 L2와 Dropout을 많이 쓴다고 하고, Dropout은 나중에 강의에서 다루게 된다고 합니다. (Dropout은 현재 특허가 걸려있어서 다른 대안책을 사용한다고 합니다. )

그 외에도 Bathch normalization 등도 많이 사용합니다.
#
![13](https://user-images.githubusercontent.com/37207332/61459050-db466600-a9a6-11e9-8f07-7ee9c6d4e78a.JPG)

L2 Regularization을 사용하여 Regularization을 하는 것을 보시면 W1와 W2가 각각 있을 때,

W1처럼 \[1,0,0,0\]이렇게 데이터 한 곳에 편향된 w보다는 w2처럼 각각의 데이터들에게 고르게 w을 부여할 수 있는

W값을 사용합니다.
#
![14](https://user-images.githubusercontent.com/37207332/61459052-db466600-a9a6-11e9-9ae8-4c5f90a47c6b.JPG)

다음으로 살펴볼 내용은 Softmax입니다.

여기서 Scores는 정규화 하지 않은 클래스를 log화한 확률입니다.

Softmax Function은 각 score값에 e(자연상수)를 취하고, 그 값들의 합으로 각각의 e를 취한 값을 나누는 것을 말합니다.

이렇게 하면 각 score값들이 클래스 분류에서의 확률 값(0.00~1.00)으로 변환되게 됩니다.
#
![15](https://user-images.githubusercontent.com/37207332/61459054-dbdefc80-a9a6-11e9-8e6b-f0287ed08d97.JPG)

그리고 그 값에 -log를 취해주는데 이유는 softmax 값을 1에 가깝게 해주기 위해서입니다.

\-log함수는 1에 가까울수록 loss가 0에 가깝게 됩니다.
#
지금까지 Sotfmax 내용을 정리해서 3개의 클래스에 대한 score를 정리해보면 다음과 같습니다.
![16](https://user-images.githubusercontent.com/37207332/61459057-dbdefc80-a9a6-11e9-80ef-9bbff9e16418.JPG)

Softmax하고 SVM을 비교해겠습니다.
![17](https://user-images.githubusercontent.com/37207332/61459058-dbdefc80-a9a6-11e9-8f12-b09bc4215945.JPG)

SVM의 경우 예를 들어서 frog나 car의 값을 조금 바꿔서 다시 계산한다고 해도

S<sub>j</sub> - S<sub>y<sub>i</sub></sub> +1에서 결국에는 정답인 클래스가 정답이 아닌 클래스+1보다 크다면 loss는 항상 0이기 때문에

정답이 아닌 클래스의 score의 변화에 대해서 둔합니다.

반면에 Softmax는 확률 기반이기 때문에 다른 클래스의 Score가 조금만 변해도

정답 클래스의 확률 값은 변하게 될 겁니다. 즉, 민감하다고 할 수 있습니다.

#
지금까지 내용을 정리해보면 아래와 같습니다.

![18](https://user-images.githubusercontent.com/37207332/61459059-dbdefc80-a9a6-11e9-916b-b43571e69201.JPG)

우리는 지금까지 dataset을 가지고 있고, Score를 계산하는 함수도 있고, 마지막으로 loss도 구할 수 있습니다.

거기에 Regularization까지 했습니다.
#
그렇다면 W값은 어떻게 조정하는 것일까요?

그 내용이 다음에 나오는 Optimization입니다.

![19](https://user-images.githubusercontent.com/37207332/61459060-dc779300-a9a6-11e9-81c2-2d212253c5b0.JPG)

cs231n 강의에서는 Optimization은 우리가 마치 산 속에서 산 아래로 하산하는 것으로 비유해서 설명합니다.

우리가 산 속에 있다면 어떻게 내려갈까요?

아무래로 우선 보이는 곳으로 한발을 내딛고, 그 다음 어디로 갈지 다시 탐색하고, 다시 발을 내딛고, 다시 다음에 갈 곳을 탐색할 겁니다.

그것과 비유해서 W값을 찾아나서는 것을 설명합니다.

먼저 Random Search입니다.

간단한 코드와 함께 설명하고 있는데, 간단합니다.

W 값을 Random하게 바꾸면서 찾는다는 건데요. 마치 우리가 산 속에서 텔레포트로 이동하는 것을 상상하시면 됩니다.

#
![20](https://user-images.githubusercontent.com/37207332/61459062-dc779300-a9a6-11e9-9a36-bb18cbc30bcb.JPG)

그렇기 때문에 당연히 성능은 좋지 않습니다. 정확도가 15.5%밖에 나오질 않네요...

최고 정확도는 95%까지 나오지만 일정한 정확도가 나오는 것이 아니기 때문에

당연히 실제로는 절대 사용하지 않습니다.

![21](https://user-images.githubusercontent.com/37207332/61459063-dc779300-a9a6-11e9-9046-d9a56ea0cd8c.JPG)
#

두 번째 방법은 경사를 따라서 이동하는 겁니다.

![22](https://user-images.githubusercontent.com/37207332/61459064-dd102980-a9a6-11e9-845a-f6aa8af0c9a4.JPG)

gradient descent 개념이 나오는데, 위에서 말씀드린 경사를 따라가는 방법입니다.

경사하강법이라고 합니다.

![23](https://user-images.githubusercontent.com/37207332/61459065-dd102980-a9a6-11e9-9cf0-74fa8f209e37.JPG)

1차원이면 미분식, 다차원이면 편미분된 벡터 값이 나옵니다.
#

구하는 방법은 아래와 같습니다.

![24](https://user-images.githubusercontent.com/37207332/61459066-dd102980-a9a6-11e9-8943-a471f3439961.JPG)
![25](https://user-images.githubusercontent.com/37207332/61459067-dd102980-a9a6-11e9-98f3-721c1833b82f.JPG)
![26](https://user-images.githubusercontent.com/37207332/61459068-dda8c000-a9a6-11e9-94dc-88863ec9403b.JPG)

각각의 W값에 0과 가까운 값을 더해서 미분값을 구하는 것입니다.

이 방법을 Numerical(수치적) gradient 방법이라고 합니다.

하지만 일일히 계산하면 데이터가 많아진다면 엄청 오래걸리겠죠?

그래서 나온 것이 해석적 방법(Analytic gradient)입니다.

![27](https://user-images.githubusercontent.com/37207332/61459070-dda8c000-a9a6-11e9-8297-76cb80b7d6e1.JPG)

멋진 두 수학자분들(뉴턴과 라이프니츠)이 만드신 거라고 하네요.

![28](https://user-images.githubusercontent.com/37207332/61459072-dda8c000-a9a6-11e9-8e9e-a6da1400d5a7.JPG)

위와 같이 값이 바로 나옵니다.
#

![29](https://user-images.githubusercontent.com/37207332/61459073-dda8c000-a9a6-11e9-9c34-92f68c510aac.JPG)

요약하자면 Numerical gradient는 approximate, slow, easy to write합니다.

정확하지 않고, 느리지만 사용하기는 쉽다고 합니다.

Analytic gradient는 exact, fast, error-prone합니다.

정확하고, 빠르지만 에러가 날 확률은 높습니다.

그래서 optimization을 할 때 일반적으로는 Analytic 방법으로 빠르게 값을 구하고,

검증할 때 Numerical을 사용해서 계산이 정확하게 이뤄지고 있는지 검토한다고 합니다.

이 방법을 gradient check라고 합니다.
