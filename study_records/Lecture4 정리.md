Lecture 4. Introduction to Neural Networks
==========================================

cs231n(Spring 2017) 강의를 정리합니다.

(본 포스팅은 cs231n 강의 Slide를 참고하여 작성하였습니다.)

강의 자료는 아래 링크를 참고하면 됩니다.

Youtube:[https://youtu.be/vT1JzLTH4G4](https://youtu.be/vT1JzLTH4G4)

Course Notes:[http://cs231n.github.io/](http://cs231n.github.io/)
#
> In Lecture 4 we progress from linear classifiers to fully-connected neural networks. We introduce the backpropagation algorithm for computing gradients and briefly discuss connections between artificial neural networks and biological neural networks.
#
![1](https://user-images.githubusercontent.com/37207332/61961469-7e7c2880-b002-11e9-887b-0150b6c7a69b.JPG)

지난 시간에 각 클래스에 대한 score와 Loss를 구했습니다.

Overfitting은 training data에 편향되어 학습이 된 것을 말합니다.

training set에는 거의 완벽하게 성능을 보이지만 test set에는 제대로된 성능이 나오지 않습니다.

마치 문제집을 매번 풀던 것만 계속 풀어서 해당 문제집에 나온 문제는 잘 풀지만

새로운 문제가 주어졌을 때는 잘 풀지 못하는 것이라고 볼 수 있습니다.

우리가 원하는 것은 새로운 이미지가 주어졌을 때 어떤 이미지인지 잘 맞추기를 원합니다.

cs231n에서는 Overfitting을 방지하는 방법으로 Regularization을 소개했습니다.

(람다 값이 너무 작으면 Overfitting이 발생할 수 있습니다.)

#
![2](https://user-images.githubusercontent.com/37207332/61961470-7e7c2880-b002-11e9-808b-9ee6a4e61f0f.JPG)

Optimization은 **Loss를 0에 가깝게 하는 hyperparameter를 찾는 것**입니다.

gradient를 이용해 Loss가 작아지도록 합니다.

#
![3](https://user-images.githubusercontent.com/37207332/61961472-7f14bf00-b002-11e9-92e4-a97eee56e02e.JPG)

gradient는 위의 수식으로 풀어서 구하는 **Numerical** 방법이 있습니다.

보통 h는 0에 근사한 값을 사용하고, 보통 1e-4(= 10<sup>-4</sup>)정도로 해서 gradient를 구합니다.

다른 방법으로는 **Analytic** 방법이 있습니다. 우리가 일반적으로 gradient를 구하는 방법으로

(x<sup>2</sup>)' = 2x 같이 푸는 방법을 말합니다.

#
![4](https://user-images.githubusercontent.com/37207332/61961474-7f14bf00-b002-11e9-9497-3a20cce28ecd.JPG)

지금까지 내용의 Computational graphs를 보면 Wx에 대해서 scores를 구하고, Loss를 구한 다음

Regularization까지 해서 최종적으로 Loss를 구하는 과정의 graph로 표현이 됩니다.

#
이제부터는 Neural Networks에 대해서 배웁니다.

![5](https://user-images.githubusercontent.com/37207332/61961475-7f14bf00-b002-11e9-9139-e2c70e603cdf.JPG)

하나의 직선(f = Wx )으로는 위와 같은 문제들을 해결할 수 없었습니다.

위와 같은 Non-Linear한 문제를 해결하는 방법은 다음과 같이 층을 쌓는 것입니다.

#
![6](https://user-images.githubusercontent.com/37207332/61961476-7fad5580-b002-11e9-8d70-8bbcfceb6072.JPG)

Layer을 쌓는 다는 것은 이전 Layer의 output이 다음 Layer의 weight와 곱해져 input으로 들어가면 됩니다.

2-Layer구조는 weight가 없는 층은 count하지 않아서 0층이고, 1층에는 0층의 output과 weight가 곱해져 input으로 들어오고, 

1층의 output과 weight가 곱해져 2층으로 가서 최종 결과가 나오게 됩니다.

max 부분은 ReLU라는 Activation function이라고 보실 수 있습니다.   

#
![7](https://user-images.githubusercontent.com/37207332/61961477-7fad5580-b002-11e9-9a7e-7c7f1d33d291.JPG)

기존에 다뤘던 CIFAR-10을 이용해서 2-Layer를 적용하면 3072의 vector가 input으로 들어가 w1과 곱해져

h(hidden Layer)의 input으로 들어가고, 다시 h-Layer에서 100개의 output으로 나와 다시 w2와 곱해져 최종적으로 10개의 클래스로 분류됩니다. 
여기서 hidden Layer의 output 갯수는 hyperparamer입니다.

#
![8](https://user-images.githubusercontent.com/37207332/61961479-8045ec00-b002-11e9-9085-e08fae00efdc.JPG)

Layer는 계속 쌓아나갈 수 있습니다.

3-layer의 경우 hidden layer가 2개입니다.

#
![9](https://user-images.githubusercontent.com/37207332/61961481-8045ec00-b002-11e9-907e-0133c8142230.JPG)

Neural Networks는 인간의 뉴런을 연구하여 만들어졌습니다.

#
![10](https://user-images.githubusercontent.com/37207332/61961483-8045ec00-b002-11e9-86c0-beb45242dfdc.JPG)

인간의 뉴런에서 자극이 들어오면 수상돌기에서 신호를 받아 추상돌기를 통해 신호를 전달합니다.

Neural Networks의 구조도 이와 비슷합니다.

입력 X(자극)이 들어오면 각 입력에 대한 강도(weight)가 곱해진 신호들의 합을 구합니다.

합에 대해서 Activation Function을 적용합니다.

임계값(Threshold)을 기준으로 신호들의 합을 Activation Function을 적용했을 때 output이 정해집니다.

#
![11](https://user-images.githubusercontent.com/37207332/61961484-80de8280-b002-11e9-92d1-bf01d8677b74.JPG)

Activation Function 몇 가지를 소개하고 있습니다.

Activation Function은 특정 무엇을 써야한다기 보다는 자신의 학습 모델에 적용할 때 다양하게 적용해보고,

가장 성능이 좋은 것을 선택하여 사용하면 됩니다.

#
![12](https://user-images.githubusercontent.com/37207332/61961487-80de8280-b002-11e9-93b1-d69808684809.JPG)

Neural Networks Architecture를 설명하고 있는데, 이전 Layer의 모든 node와 다음 Layer의 모든 node가 연결된 것을

Fully-connected Layer라고 부릅니다.

#
![13](https://user-images.githubusercontent.com/37207332/61961488-80de8280-b002-11e9-8984-337b044b1cde.JPG)

지금까지 배웠던 Loss를 구하는 과정을 Forward pass 또는 Forward propagation이라고 합니다.

그와 반대로 Loss쪽에서 input쪽으로 가면서 gradient를 구하는 것을 Backpropagation이라고 합니다.

#
![14](https://user-images.githubusercontent.com/37207332/61961491-80de8280-b002-11e9-8103-97a49f9cb7a3.JPG)

Backward propagation (backpropagation)은 Loss쪽에서 input쪽으로 gradient를 계산하면서

input이 Loss에 미치는 영향을 구할 수 있습니다.

그래서 최종적으로는 Loss를 0에 가깝게 Optimization을 해야하기 때문에

backpropagation을 통해서 input이 Loss을 크게 한다면 input값이 감소하도록 hyperparameter를 조정할 수 있고,

반대로 input이 Loss 값을 0보다 작게 한다면 input값이 증가하도록 hyperparameter를 조정할 수 있습니다.

#

Backpropagation하는 것을 간단한 예제를 통해 설명합니다.

![15](https://user-images.githubusercontent.com/37207332/61961492-81771900-b002-11e9-9ee1-1fe235f42fa9.JPG)

x+y 의 결과를 q로 놓고, q\*z를 f로 놓았습니다.

q = x+y, f=qz

각 변수에 대해서 편미분을 구한 식이 빨간 부분과 파란 부분입니다.

최종적으로 구하고자 하는 것은 x,y,z값의 f(loss)에 대한 편미분, 즉 x,y,z값이 f에 미치는 영향입니다.

#
![16](https://user-images.githubusercontent.com/37207332/61961493-81771900-b002-11e9-8696-63afb139ee9f.JPG)

df/df는 자신의 대한 gradient이므로 1입니다. 즉, Loss의 gradient는 1입니다.

#
![17](https://user-images.githubusercontent.com/37207332/61961494-81771900-b002-11e9-92a5-ecce36196128.JPG)

다음으로 df/dz를 구하면 파란색 부분에서 구해놨던 식에 의해서 q가 됩니다. 따라서 값은 3입니다.

즉, z값이 a만큼 변한다면 f값은 3a만큼 변한다고 할 수 있습니다.

#
![18](https://user-images.githubusercontent.com/37207332/61961495-820faf80-b002-11e9-8c30-4f13645bb617.JPG)

df/dq도 마찬가지로 파란색 부분에서 구해놨던 식에 의해서 z가 됩니다. 따라서 값은 -4입니다.

#
![19](https://user-images.githubusercontent.com/37207332/61961497-820faf80-b002-11e9-9d85-34610bb9060f.JPG)

다음으로는 df/dy를 구해야하는데 df/dy는 Chain rule을 적용해서 구할 수 있습니다.

따라서 1\*z가 되어 -4 값이 나옵니다.

#
![21](https://user-images.githubusercontent.com/37207332/61961499-820faf80-b002-11e9-9f9d-792f9f4b8e7f.JPG)

x값도 마찬가지로 Chain rule을 적용해서 구하면 1\*z이므로 -4가 나옵니다.

이렇게 구하고자 하는 x, y, z의 f에 대한 gradient를 구했습니다.

정리하면 'x,y,z값이 a,b,c만큼 각각 변한다면 '-4a, -4b, 3c만큼 f에 영향을 준다.' 또는 '\-4a, -4b, 3c만큼 f값이 변한다' 라고 할 수 있습니다.

#
![22](https://user-images.githubusercontent.com/37207332/61961501-82a84600-b002-11e9-87db-3622e1be951f.JPG)

Forward propagation 과정에서 input에 대한 gradient(local gradient)를 구할 수 있고,

backpropagation 과정에서 gradients(global gradient)를 구할 수 있습니다.

Chain rule을 이용하면 (global)gradient \* local gradient를 통해 input 값의 gradient를 구할 수 있습니다.

#
![23](https://user-images.githubusercontent.com/37207332/61961503-82a84600-b002-11e9-91ae-6bc61d9fc11b.JPG)

x, y 앞에 node가 있어도 마찬가지로 Chain rule을 이용해서 gradient를 구할 수 있습니다.

#
다른 예제 통해 backpropagation을 설명합니다.

이번 예제는 (w0x0 + w1x1 + w2)의 계산 값이 Sigmoid function에 input이 되어 계산되는 graph입니다.

![24](https://user-images.githubusercontent.com/37207332/61961504-82a84600-b002-11e9-9f62-c6c5305a6f3b.JPG)
#

![25](https://user-images.githubusercontent.com/37207332/61961506-82a84600-b002-11e9-8723-5a970fe4a4e5.JPG)

최종 값의 gradient는 앞서 배웠던 것처럼 1이 됩니다.

#
![26](https://user-images.githubusercontent.com/37207332/61961507-8340dc80-b002-11e9-8bcf-daa61cde6a75.JPG)

지금부터는 Chain rule을 적용하여 backpropagation을 구합니다.

다음으로 1/x부분의 gradient는 -1/x^2이 되고, input으로는 1.37이 들어와서 연산이 됩니다.
#

global gradient는 1이 되고, Chain rule에 의해 두 값을 계산하면 다음과 같이 나옵니다.

![27](https://user-images.githubusercontent.com/37207332/61961508-8340dc80-b002-11e9-9f36-73acb4326fa2.JPG)

#
같은 방법으로 -0.53값과 +1 노드를 계산하면 다음과 같습니다.

![28](https://user-images.githubusercontent.com/37207332/61961509-8340dc80-b002-11e9-92d5-36a3b1795bfc.JPG)

#
exp 노드에 대한 연산도 같은 방법으로 진행해줍니다.

단순하게 이전의 노드의 gradient와 다음 노드의 gradient 값을 곱하기만 해주면 됩니다.

![29](https://user-images.githubusercontent.com/37207332/61961512-83d97300-b002-11e9-9eda-4d50f1e71a5d.JPG)
![30](https://user-images.githubusercontent.com/37207332/61961513-83d97300-b002-11e9-8552-7e049b3a13ef.JPG)
![31](https://user-images.githubusercontent.com/37207332/61961514-83d97300-b002-11e9-9f7c-368075c201aa.JPG)

#
\+ 노드는 앞서 배웠던 x+y 규칙과 같으니 gradient 값이 그대로 나가게 됩니다.

![32](https://user-images.githubusercontent.com/37207332/61961517-84720980-b002-11e9-99f7-866d256549a7.JPG)

#
\* 노드 또한 앞서 배운 규칙에 의해서 값이 구해집니다.

![33](https://user-images.githubusercontent.com/37207332/61961518-84720980-b002-11e9-9091-c21fedfe5074.JPG)

![34](https://user-images.githubusercontent.com/37207332/61961519-84720980-b002-11e9-9f58-ba462b255f1a.JPG)
#

이번 예제는 어떤 input(w0x0 + w1x1 + w2)가 sigmoid의 input으로 들어간다고 했습니다.

그래서 다음과 같이 sigmoid에 대한 gradient를 구하면 \*-1, exp, +1, 1/x  4단계를 거쳐 gradient를 구하지 않고,

sigmoid에 대한 gradient를 구해서 0.20이라는 값을 구할 수 있습니다.

![35](https://user-images.githubusercontent.com/37207332/61961521-84720980-b002-11e9-9842-c976ff6e770a.JPG)
#

연산 노드에 대한 패턴이 몇 가지 있습니다.

+노드의 경우 이전의 gradient 값이 그대로 나가게 되서 distributor라고 합니다.

\* 노드의 경우 이전의 gradient 값과 반대쪽 input 값이 곱해져 gradient가 구해져서 switcher라고 합니다.

max 노드는 input 값들 중 가장 큰 값을 제외한 나머지는 gradient가 0이 됩니다.

![36](https://user-images.githubusercontent.com/37207332/61961522-850aa000-b002-11e9-9e46-07e12869c1d0.JPG)
#

만약 gradient가 복수 개가 들어있다면 모든 노드의 gradient를 더해주면 된다고 합니다.

![37](https://user-images.githubusercontent.com/37207332/61961524-850aa000-b002-11e9-8e11-5e27219a870f.JPG)
#

local gradient에 대한 값들을 Jacobian matrix를 이용해 미리 구해놓으면 연산이 편해진다고 합니다.

![39](https://user-images.githubusercontent.com/37207332/61961525-850aa000-b002-11e9-980c-428f19f62ad6.JPG)
#

만약 input으로 4096 dimension의 data가 들어오고 4096 dimension의 output이 나간다면 

Jacobian matrix는 4096x4096이 됩니다.

또한 mini-batch를 적용하면 batch 크기 배수 만큼의 Jacobian matrix가 된다고 합니다.

![40](https://user-images.githubusercontent.com/37207332/61961526-85a33680-b002-11e9-87ce-c04b696cd2a0.JPG)
#

지금까지는 input 값이 단일 값이였다면 이번에는 input이 matrix일 때의 Forward pass와 backpropagation을 구합니다. 

![1](https://user-images.githubusercontent.com/37207332/61963677-2267d300-b007-11e9-9cb5-e41752b7a17c.JPG)
#

W, X(matrix이기 때문에 대문자로 표시하겠습니다.) WX는 구하면 q가 나옵니다. (2x2)●(2x1)=(2x1)

![2](https://user-images.githubusercontent.com/37207332/61963678-2267d300-b007-11e9-9756-010f41903c23.JPG)
![3](https://user-images.githubusercontent.com/37207332/61963679-2267d300-b007-11e9-96da-e942caeab8a8.JPG)

q^2 미분 값은 2q가 나옵니다. 따라서 df/dq 값이 위와 같이 나온 것입니다.
#

![4](https://user-images.githubusercontent.com/37207332/61963681-23006980-b007-11e9-8dab-6b0984bea930.JPG)

다음은 df/dw를 구해야하는데 Chain rule에 의해서 df/dq \* dq/df가 됩니다.

df/dq는 앞서 구했던 2q이고, dq/dw는 (1 k=i) \* Xj 이기 때문에 df/dw는 2qx가 됩니다.
#

q와 x가 2x1, 2x1이므로 하나를 Transpose를 해줘서 계산합니다.

![5](https://user-images.githubusercontent.com/37207332/61963682-23006980-b007-11e9-9bdc-f88a33b5ff6b.JPG)


빨간 부분은 항상 결과 값의 shape를 주의하라는 건데, 입력된 변수와 항상 같은 shape으로 나오는지 확인해야 합니다.
#

![6](https://user-images.githubusercontent.com/37207332/61963685-23006980-b007-11e9-8d94-a2e283c8e1c3.JPG)

X값도 마찬가지로 gradient를 구하면 위와 같이 나옵니다.

matrix 연산이라 조금 복잡했지만 gradient와 chain rule을 적용해 푸는 것은 기존에 풀었던 것과 비슷합니다. 
#

![7](https://user-images.githubusercontent.com/37207332/61963686-23006980-b007-11e9-98cf-8d9b102acaff.JPG)

Lecture4까지 Forward를 통해서 input 값들에 대해서 Loss를 구했었고,

backpropagation을 통해서 각 input 값들이 Loss에 얼만큼의 영향을 주고 있는지를 구했습니다.
