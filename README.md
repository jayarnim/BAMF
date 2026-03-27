# Bayesian Attentional Matrix Factorization

- 발표일자: 2024.11.1.

- 제1저자: [`Wang,J.`](https://github.com/jayarnim)

- 교신저자: [`Lee,J.`](https://github.com/jaylee07)

## 개요

암시적 피드백은 사용자의 직접적인 선호 표현이 아니라, 관찰 가능한 행동을 나타낸다. 따라서 이는 본질적으로 사용자 선호를 나타내는 신호로서 모호성을 가진다. 본 연구는 이 암시적 피드백의 모호성을 사용자 선호에 대한 인식적 불확실성 문제로 재해석하고, 이를 베이지안 프레임워크를 통해 모델링하는 잠재요인 모형을 제안한다.

구체적으로 어텐션 메커니즘을 적용하여 매칭 대상이 되는 사용자 혹은 아이템의 상호작용 이력을 가중합하여 새로운 벡터 표현을 생성한다. 질의 정보로는 매칭 대상이 되는 사용자와 아이템의 임베딩 벡터를 활용한다. 이때 어텐션 스코어를 선호 점수(혹은 선호 관점에서의 유사도 점수)라 가정한다면, 이 새로운 벡터 표현은 사용자의 상호작용 이력 안에서 재구성된 선호 표현이라 간주될 수 있다.

그러나 어텐션 메커니즘의 입력값으로 사용되는 벡터들은 암시적 피드백 데이터를 통해 학습되었다는 점에서 선호 신호를 반영한다고 볼 수 없다. 때문에 이들 벡터 간 어텐션 스코어를 선호 점수라 확정할 수 없다. 이 불확실성은 다수의 준거 집합에서 발생하는 잡음이 아니라, 단일 준거를 선호라 신뢰할 수 없는 점에서 기인한다. 따라서 어텐션 스코어를 확률변수로 대체하고, 전역 선호 점수를 사전 정보로 활용하여 인식적 불확실성을 반영한다.

## 아키텍처

![01](/desc/model.png)

제안 모형에서 엔티티의 잠재 표현은 두 가지 용도로 사용되고 있다. 매칭 대상일 때의 잠재 표현은 사용자-아이템 상호작용에서 관측되는 행동 신호를 나타내고, 상호작용 이력일 때의 잠재 표현은 엔티티의 선호를 형성하는 맥락 정보를 나타낸다. 초록에서는 임베딩 행렬을 용도에 따라 각각 생성하지 않고, 하나의 임베딩 행렬을 선형 변환하고 있다. 이는 행동 신호와 맥락 정보가 동일한 잠재요인에 기반하지 아니하고 서로 다른 기저 공간에서 해석되어야 함을 전제한다. 하지만 본 레파지토리에서는 용도에 따라 임베딩 행렬을 각각 생성하는 방향으로 조정하였다. 이는 행동 표현과 선호 맥락이 동일한 기저를 공유하면서 제공하는 정보가 다른 표현임을 전제한다.

제안 모형은 (1) 어느 엔티티의 상호작용 이력을 활용할 것인지, (2) 행동 표현과 선호 표현을 어떻게 결합할 것인지에 따라 구성 전략이 세분화될 수 있다. 우선, 두 엔티티의 상호작용 이력을 모두 활용하는 경우, 사용자의 선호 표현은 해당 사용자가 상호작용한 아이템을 선호도로 가중합한 표현이 된다. 마찬가지로 아이템의 선호 표현은 해당 아이템이 상호작용한 사용자를 선호도로 가중합한 표현이 된다. 이에 반해 둘 중 한 엔티티의 상호작용 이력만을 활용하는 경우, 이력이 활용된 엔티티의 선호 표현은 마찬가지로 이력을 선호도로 가중합한 표현이다. 하지만 이력이 활용되지 않은 엔티티의 선호 표현은 선호 유사도로 가중합한 표현이 되며, 이는 사용자 기반 협업 필터링 혹은 아이템 기반 협업 필터링과 동일하다. 초록에서는 사용자 이력만을 활용하였으나, 본 레파지토리에서는 세 가지 전략을 모두 실험하였다.

한편, 행동 표현과 선호 표현의 결합 전략과 선형적 의미는 다음과 같다. 아래 전략들은 DNCF(He et al., 2021)에서 제안된 함수들이다. 단, 초록에서는 행동 표현과 선호 표현을 결합하지 않고 선호 표현만을 사용하였다.

- `sum`: 두 벡터의 신호를 누적하여 새로운 벡터로 합성하는 정보 누적 연산
- `att`: 두 벡터의 신호를 일정 비율로 반영하여 두 벡터 사이 벡터로 보간하는 정보 선택 연산
- `mean`: 두 벡터의 신호를 균등 비율로 반영하여 두 벡터 사이 벡터로 보간하는 정보 선택 연산
- `prod`: 두 벡터의 신호를 요소별로 결합하여 공통으로 활성화된 성분을 강조하는 정보 여과(상호작용) 연산
- `cat`: 후속 레이어(e.g. mlp, linear, etc.)에서 정보 누적과 선택, 여과를 수행하기 위하여 두 벡터의 정보를 보존함

## 표기

### idx

- $u=0,1,2,\cdots,M-1$: target user
- $i=0,1,2,\cdots,N-1$: target item
- $v \in R_{i}^{+} \setminus \{u\}$: history users of target item (target user $u$ is excluded)
- $j \in R_{u}^{+} \setminus \{i\}$: history items of target user (target item $i$ is excluded)
- $M,N$ is padding idx

### vector

- $p \in \mathbb{R}^{M \times K}$: user id embedding vector (we define it as global behavior representation)
- $q \in \mathbb{R}^{N \times K}$: target item id embedding vector (we define it as global behavior representation)
- $\phi \in \mathbb{R}^{M \times K}$: history user id embedding vector
- $\psi \in \mathbb{R}^{N \times K}$: history item id embedding vector
- $c_{u} \in \mathbb{R}^{M \times K}$: user context vector (we define it as conditional preference representation)
- $c_{i} \in \mathbb{R}^{N \times K}$: item context vector (we define it as conditional preference representation)
- $z_{u} \in \mathbb{R}^{M \times K}$: user refined representation
- $z_{i} \in \mathbb{R}^{N \times K}$: item refined representation
- $z_{u,i}$: $(u,i)$ pair predictive vector
- $x_{u,i}$: $(u,i)$ pair interaction logit
- $y_{u,i}, \hat{y}_{u,i}$: $(u,i)$ pair interaction probability

### function

- $\mathrm{bam}(q,k,v)$: bayesian attention module (only single head)
- $\mathrm{comb}(\cdot)$: behavior rep & preference rep combination function (e.g. `sum`, `att`, `mean`, `prod`, `cat`)
- $\odot$: element-wise product
- $\oplus$: vector concatenation
- $\mathrm{ReLU}$: activation function, ReLU
- $\sigma$: activation function, sigmoid
- $W$: linear transformation matrix
- $h$: linear trainsformation vector
- $b$: bias term

## 모형

### 사용자 표현

- user global behavior:

$$
p_{u}=\mathrm{embedding}(u)
$$

- user conditional preference:

$$
c_{u}=\mathrm{bam}(p_{u}, \forall \psi_{j}, \forall \psi_{j})
$$

- user refined representation:

$$
z_{u}=\mathrm{comb}(p_{u}, c_{u})
$$

### 아이템 표현

- item global behavior:

$$
q_{i}=\mathrm{embedding}(i)
$$

- item conditional preference:

$$
c_{i}=\mathrm{bam}(q_{i}, \forall \phi_{v}, \forall \phi_{v})
$$

- item refined representation:

$$
z_{i}=\mathrm{comb}(q_{i}, c_{i})
$$

### 매칭 함수

- element-wise product (agg & matching):

$$
z_{u,i}=z_{u} \odot z_{i}
$$

- logit:

$$
x_{u,i}=h^{T}(W \cdot z_{u,i}+b)
$$

- prediction:

$$
\hat{y}_{u,i}=\sigma(x_{u,i})
$$

### 목적 함수

$$
\mathcal{L}_{\mathrm{ELBO}}:= \sum_{(u,i)\in\Omega}{\left(\mathrm{NLL} + \sum_{j \in R_{u}^{+} \setminus \{i\}}{\mathrm{KL}^{(u,j)}} + \sum_{j \in R_{u}^{+} \setminus \{i\}}{\mathrm{KL}^{(i,j)}} \right)}
$$

- apply `bce` to pointwise `nll`:

$$
\mathcal{L}_{\mathrm{BCE}}:=-\sum_{(u,i)\in\Omega}{y_{u,i}\ln{\hat{y}_{u,i}} + (1-y_{u,i})\ln{(1-\hat{y}_{u,i})}}
$$

- apply `bpr` to pairwise `nll` (only log-likelihood):

$$
\mathcal{L}_{\mathrm{BPR}}:=-\sum_{(u,pos,neg)\in\Omega}{\ln{\sigma(x_{u,pos} - x_{u,neg})}}
$$

## 실험

제안 모형의 성능을 측정하기 위하여 다음의 데이터 셋을 활용하였다:

- movielens latest small (2018) [`link`](https://grouplens.org/datasets/movielens/latest/)

데이터 셋을 `trn`, `val`, `tst` 각각 8:1:1 로 사용자 기준 계층적 분할하였다. 추가로 `leave-one-out` 데이터 셋을 구성하여 학습 조기종료 시점을 모니터링하는데 사용하였다. `opt`(`trn`, `val`) 데이터 셋에 대하여 1:4, `msr`(`tst`, `loo`) 데이터 셋에 대하여 1:99 비율로 네거티브 샘플링을 적용하였다. 초기 10 epoch 에 대해서는 조기종료 여부를 모니터링하지 않았고, 11 epoch 부터 모니터링하여 `ndcg@10` 이 최대 5회 개선되지 않을 경우 학습을 조기종료하였다.

활용된 데이터 셋에서는 일부 사용자의 상호작용 이력이 2,000 건이 넘는데 반해, 상위 10% 사용자의 이력은 400 건이다. 효율성을 도모하기 위하여 T선별 점수 기준 상위 400 건의 이력만을 활용하였다. 선별 점수로는 상호작용 빈도와 TF-IDF 를 활용하였으며, TF-IDF 를 활용하였을 때 추가적인 성능 개선이 있었다. TF-IDF 를 활용할 때는 문서를 상호작용 이력에, 단어(혹은 토큰)를 해당 이력의 구성자에 대응하여 점수를 산출하였다.

초록에서는 어텐션 스코어 함수로서 NAIS(He et al., 2018)에서 제안한 함수들을 적용하였으나, 본 레파지토리에서는 내적을 적용하여 실험을 진행하였다. 또한 본 레파지토리에서는 어텐션 스코어의 사전 분포와 변분 분포로서 로그 정규 분포를 활용하였다. 이때 사전 분포의 표준편차는 1.0, 변분 분포의 표준편차는 0.1 로 임의 고정하였다. 다만, 해당 수치는 최적값이라 볼 순 없으며 데이터 셋에 따라 조정될 수 있다.

실험 결과, 두 엔티티의 상호작용 이력을 모두 활용하였을 때(bimodal) 성능이 가장 우수하였으며, 한 엔티티의 이력만 활용할 때는 사용자 이력을 활용하였을 때가 아이템 이력을 활용하였을 때보다 우수하였다. 결합 전략의 경우, 다섯 전략 중 `sum` 이 대체로 높은 성능을 보장하였다. 다만, 행동 표현과 선호 표현을 결합할 때보다 선호 표현만 사용하였을 때 성능이 더 높았다:

- combined behavior & preference
    - applied user, item histories [`notebook`](/notebooks/comb_bimodal/)
    - applied user histories [`notebook`](/notebooks/comb_user/)
    - applied item histories [`notebook`](/notebooks/comb_item/)

- only used preference
    - applied user, item histories [`notebook`](/notebooks/context_bimodal/)
    - applied user histories [`notebook`](/notebooks/context_user/)
    - applied item histories [`notebook`](/notebooks/context_user/)