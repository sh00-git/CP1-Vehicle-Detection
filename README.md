## CP1 : YOLO 모델을 이용한 차량 인식 시스템

- 개발기간
    - 2022.11.02 ~ 2022.11.13
- 사용 언어 및 라이브러리
    - `Python`, `Pytorch`

## 💡 Topic

- 객체 탐지에 사용되는 모델인 YOLO에 대해 학습하고 YOLO 모델을 사용하여 차량을 인식하는 모델을 구현하는 프로젝트
- AI 부트캠프 CP1

## ❓ 문제정의

본 프로젝트를 통해 객체 탐지에 사용되는 모델인 YOLO에 대해 학습하고 데이터셋을 모델에 맞게 전처리하는 과정 및 모델 학습 과정을 익힌다. 또한, 차량 인식을 시작으로 추후의 다른 서비스를 개발 및 접목시키기 위한 발판을 마련한다.

## 🔍 프로젝트 진행과정

1. 사전 기획
2. YOLO 원리 및 개념 학습
3. 데이터 수집
4. 모델링 및 데이터 전처리

## 📚 데이터 셋

1. Vehicles-OpenImages Dataset
- 출처 : roboflow
- 크기 : 416 x 416 크기의 데이터 627개
2. Road Vehicle Images Dataset
- 출처 : kaggle
- 크기 : 640 x 426 크기의 데이터 3004개
3.  차량 및 사람 인지 영상 데이터
- 출처 : AI Hub
- 크기 : 1920 x 1080 크기의 120만개

## ✍🏻 프로젝트 수행 과정 및 결과

1. 논문 “You Only Look Once: Unified, Real-Time Object Detection”을 통한 YOLO 내용을 학습
    - 객체 탐지란 무엇인지
        - 이미지에서 객체와 그 경계 상자(Bounding box)를 탐지
        - 객체 탐지 = 분류 + 위치 정보
    - YOLO의 장단점
        - Yolo : 이미지 전체에 대해서 하나의 신경망이 한 번의 계산만으로 bounding box와 클래스 확률을 예측
        - 장점: 빠른 속도, 예측시 이미지 전체를 봄, 검출 정확도가 높음
        - 단점: 작은 물체에 대한 검출 정확도가 낮음, 속도와 정확성이 trade-off 관계
    - YOLO 구조
        - 앞단은 이미지로부터 특징을 추출하는 Convolutional layer, 이어서 클래스 확률과 Bounding box의 좌표(coordinates)를 예측하는 fully-connected layer로 구성
        - 24개의 convolutional layer와 2개의 fully-connected layer
        - Bounding box는 5개의 예측치로 구성 (x, y, w, h, confidence)
            - (x, y): bounding box 중심의 그리드 셀 내 상대 위치
            (w, h): bounding box의 상대 너비와 상대 높이
            Confidence: bounding box가 객체를 포함한다는 것을 얼마나 믿을 만한지, 예측한 bounding box가 얼마나 정확한지를 나타냄
    - YOLO 학습 과정
        1. ImageNet 데이터 셋으로 YOLO의 앞 단 20개의 Convolutional layer 을 사전 훈련시킨다.
        2. 사전 훈련된 20개의 Convolutional layer 뒤에 4개의 Convolutional layer 및 2개의 fully-connected layer 을 추가한다.
        3. YOLO 신경망의 마지막 계층에는 선형 활성화 함수(linear activation function)를 적용하고, 나머지 모든 계층에는 leaky ReLU를 적용한다.
            - 활성화 함수 : 입력 신호의 총합을 출력 신호로 변환하는 함수
            Leaky ReLU : 0 이하의 값도 작은 음수 값을 가짐
        4. 과적합(overfitting)을 막기 위해 드롭아웃(dropout)과 data augmentation을 적용한다.
2. 프로젝트에서 사용할 YOLO 버전 선택
    - YOLOv5 모델 선정 이유
        - YOLOv1, v2는 정확도가 낮다는 문제가 있고 v6, v7은 비교적 최근에 나와서 정보가 부족하고 최적화를 고려하여 선택하지 않음. v4에 비해 v5는 낮은 용량과 빠른 속도를 가짐
        - YOLOv3와 YOLOv5 중 YOLOv5는 이미지로부터 Feature map을 추출하는 부분인 Backbone이 다양하여 상황에 맞게 선택해서 사용할수 있는 장점이 있다고 판단하여 YOLOv5를 선택하였다.
3. Roboflow 데이터 사용
    - Class 수: 5개
    - Train : test : valid = 878 :126 : 250
    - 예측결과
        
        <img src="https://user-images.githubusercontent.com/76083173/212466388-e24d501a-a225-4625-ae1e-e406e3b7e04d.png" height="300" />
        
    - 평가지표
        - valid
            
            <img src="https://user-images.githubusercontent.com/76083173/212466404-f0020056-b41b-4211-8a32-77b1ec11d2bb.png" height="100" />
            
        - test
            
            <img src="https://user-images.githubusercontent.com/76083173/212466428-2dd90116-4693-4970-b2e4-dab678e600e8.png" height="100" />
            
    - 결과
        - valid는 maP값이 0.367, test는 maP값이 0.5로 높지 않음.
        - precision과 recall도 대체적으로 0.5보다 적게 나오는 모습을 보임.
        - 이미지의 수가 적고 데이터가 대부분 Car 클래스에 편향되어 있어 정확도에 영향을 미친 것을 보임.
            
            <img src="https://user-images.githubusercontent.com/76083173/212466456-ed4196a8-74b7-462c-9e31-466576ba75fc.png" height="100" />
            
4. kaggle 데이터 사용
    - Class 수 : 21개
    - Train : valid = 2704 : 300
    - 예측결과
        
        <img src="https://user-images.githubusercontent.com/76083173/212466477-293307a5-26a1-42df-8f9a-40845f95e2b4.png" height="400" />
        
    - 평가지표
        
        <img src="https://user-images.githubusercontent.com/76083173/212466505-d85b4ed1-3420-4fc9-a4bc-56a30675670e.png" height="250" />
        
    - 결과
        - maP값이 0.366으로 낮게 나옴.
        - precision과 recall도 대체적으로 0.5보다 적게 나오는 모습을 보이고, precision이 1, recall이 0으로 나오는 클래스가 있는 것으로 보아 데이터의 문제가 있다는 것으로 생각됨.
        - 데이터의 문제와 너무 많은 클래스의 수가 정확도에 영향을 미친것으로 생각됨.
5. AI Hub 데이터 사용
    - yolo 모델에 맞게 데이터 형식 변경
        - AI HUB 데이터의 바운딩박스 형식
            - 바운딩 박스의 네 꼭지점 좌표
        - YOLO 데이터의 바운딩박스 형식
            - class, x, y, w,h
            - x, y : 바운딩박스 중심점, 그리드 셀의 범위에 대한 상대값 입력
            - w, h : 전체 이미지대비 바운딩 박스의 상대값(너비, 높이)
    - YOLO 데이터셋 폴더 구조에 맞게 데이터 셋 구축
        - 영상을 이미지로 저장해 놓은 데이터로 순차적으로 들어가 있을 시 학습에 영향을 줄 수 있다고 판단하여 이미지를 랜덤으로 섞음.
        - Train : valid : test = 7 : 2 : 1

## ✨ Learned
- 분류와 `객체 탐지`의 차이점 및 `Yolo의 원리 및 개념`, 구성에 대해 배울 수 있었다.
- Pre-traine된 `YOLO 모델`을 사용하는 방법을 익힐 수 있었다.
- `YOLO v5 모델`을 custom data를 사용하여 학습하는 flow를 익힐 수 있었다.
- `YOLO 모델에 맞게 데이터를 전처리`하는 과정을 수행할 수 있었다.
- 데이터를 선정할 때 `클래스와 데이터의 수를 고려해야할 필요성`을 느낌.
    - 데이터의 편향이 보일때의 대처법을 떠올리지 못함.
        - 데이터 증강 및 데이터 추가 적재 등 방법을 통하여 데이터를 보안하는 것의 필요성을 느낌.
