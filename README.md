# OSS Personal Project


# 지원 Operating Systems

Docker를 이용하여 실행하는 프로젝트이므로, 다음 운영 체제에서 지원됩니다:

| OS      | 지원 여부 |
|---------|-----------|
| Windows | ⭕         |
| Linux   | ⭕         |
| MacOS   | ⭕         |


# **인공지능 기반 심장병 예측 모델 웹사이트**

심장병에는 여러 유형이 있지만, 대부분의 사람들은 흉통, 심장마비, 또는 갑작스러운 심장 정지와 같은 증상이 나타난 후에야 자신이 이 질병을 가지고 있다는 것을 알게 됩니다. 심장마비와 같은 부정적인 결과가 발생하기 전에 심장 질환을 정확하게 예측할 수 있는 예방 조치가 필요합니다. 고혈압, 고혈중 콜레스테롤, 흡연이 심장 질환의 세 가지 주요 위험 요소인데, 이를 포함한 7가지 수치를 사용하여 심장병 위험을 감지할 수 있습니다.



*Trained Model Accuracy: 0.9032910100673116* <br/>
제가 학습한 모델의 정확도는 **90.33%** 입니다. (heart_disease_model.pth)



### About the dataset:
Behavioral Risk Factor Surveillance System (BRFSS)는 미국 질병통제예방센터(CDC)가 매년 실시하는 건강 관련 전화 설문 조사입니다. 가져온 데이터셋은 심장병의 이진 분류를 위해 사용될 253,680개의 정리된 BRFSS 2015 설문 응답을 사용합니다. 원본 데이터셋의 330개의 특징 중, 7개만을 사용했습니다.

**사용한 features**:
1. High Blood Pressure: Have you EVER been told by a doctor, nurse or other health professional that your blood pressure is high?
2. High Cholesterol: Have you EVER been told by a doctor, nurse or other health professional that your blood cholesterol is high?
3. BMI 지수 (Body Mass Index)
4. Smoking: Have you smoked at least 100 cigarettes in your entire life?
5. Fruit Consumption: Do you consume fruit 1 or more times per day?
6. Vegetable Consumption: Do you consume vegetables 1 or more times per day?
7. Alcohol Consumption: Are you a heavy drinker?



## 실행 방법
이 프로젝트는 Docker를 사용하여 운영 체제에 관계없이 일관된 환경에서 실행할 수 있습니다. 아래는 Windows, Linux, MacOS에서 웹사이트를 실행하는 방법입니다.

#### Windows, Linux, MacOS 공통 실행 방법
1. **Docker 설치**
    - Docker 공식 웹사이트에서 Docker를 설치합니다.

2. **프로젝트 클론**
    ```sh
    git clone <repository_url>
    cd <repository_directory>
    ```

3. **Docker 이미지 빌드**
    ```sh
    docker build -t disease_prediction .
    ```

4. **Docker 컨테이너 실행**
    ```sh
    docker run -it --name webserver -p 8000:8000 disease_prediction
    ```

5. **웹사이트 접속**
    - 웹 브라우저를 열고 [http://localhost:8000/](http://localhost:8000/)에 접속합니다.


#### 추가 설명
- Docker를 사용하여 실행되므로, Docker가 설치되어 있으면 Windows, Linux, MacOS에서 동일한 방법으로 실행할 수 있습니다.

- 이미 사용 중인 포트가 있는 경우, `-p <host_port>:8000` 형식으로 다른 포트를 지정하여 실행할 수 있습니다.
    ```sh
    docker run -it --name webserver -p 8080:8000 disease_prediction
    ```
  그런 다음 [http://localhost:8080/](http://localhost:8080/)에 접속합니다.


## 실행 예시

#### 웹사이트 첫 화면입니다.
![image](https://github.com/yjchoco/oss_personal_project_phase_1/assets/105093937/22bc27e7-a9af-479a-b75e-8c9ec02c37ec)

#### 해당하는 항목 체크한 후, predict를 누르면,
![image](https://github.com/yjchoco/oss_personal_project_phase_1/assets/105093937/b6dc1619-87e7-41bf-8901-272001eeec62)

#### predition 결과가 출력됩니다.
![image](https://github.com/yjchoco/oss_personal_project_phase_1/assets/105093937/fd644f83-4a4a-4384-a76e-c07bde19055d)


## **코드 설명**
------

## *main.py*
이 main.py 파일은 FastAPI를 사용하여 간단한 웹 애플리케이션을 구축합니다.

##### DiseasePredictor 클래스 
이 클래스는 심장 질환 예측을 위한 인공 신경망 모델을 정의합니다.
DiseasePredictor 모델 인스턴스를 생성하고 저장된 가중치(heart_disease_model.pth)를 로드합니다.

##### predict 함수
predict 함수는 데이터를 입력 받아 모델을 사용하여 예측을 수행합니다.


FastAPI 애플리케이션을 초기화하고 템플릿 디렉토리를 설정한 뒤, 사용자가 입력한 폼 데이터를 받아 심장 질환 예측을 수행합니다. 예측 결과에 따라 "Low risk of disease" 또는 "High risk of disease" 메시지를 생성합니다. 사용자가 입력한 폼 데이터를 기반으로 심장 질환 발생 가능성을 예측하는 간단한 웹 애플리케이션을 구축합니다.

-------

## *train.py*
모델을 학습하여 훈련된 모델의 가중치를 저장할때 필요한 코드 파일로, 웹사이트를 실행할 때에는 필요하지 않습니다.

##### DiseasePredictor 클래스
이 클래스는 심장 질환 예측을 위한 인공 신경망 모델을 정의합니다.
3개의 선형 계층과 dropout 계층이 포함된 신경망을 정의하고, forward 함수를 통해 입력 데이터를 순전파시킬 수 있습니다.

##### HeartDiseaseDataset 클래스
이 클래스는 데이터셋을 정의합니다. 데이터셋을 초기화하고, 입력 데이터(X)와 레이블(y)을 텐서로 변환합니다.

##### evaluate_model 함수
이 함수는 테스트 데이터셋을 사용하여 모델을 평가합니다.
모델을 eval 모드로 설정하고, 테스트 데이터셋을 통해 예측을 수행한 후, 정확도를 계산하여 반환합니다.

##### 전체 코드 흐름
1. **데이터 로드 및 전처리**
    - CSV 파일에서 데이터를 로드하고, 데이터를 train set와 test set으로 분할합니다.

2. **데이터셋 및 데이터로더 생성**
    - `HeartDiseaseDataset` 클래스를 사용하여 훈련 데이터와 테스트 데이터셋을 생성합니다.
    - `DataLoader`를 사용하여 배치 단위로 데이터를 로드할 수 있도록 합니다.

3. **모델, 손실 함수, 최적화 도구 초기화**
    - `DiseasePredictor` 모델을 생성하고,
    - 손실함수로는 cross-entropy loss, 옵티마이저로는 Adam을 사용합니다.

4. **모델 훈련**
    - 지정된 에포크 동안 훈련 데이터를 사용하여 모델을 훈련시킵니다.
    - 각 에포크 후 테스트 데이터셋을 사용하여 모델의 정확도를 평가합니다.

5. **모델 저장**
    - 훈련된 모델의 가중치를 저장합니다.

-------

### requirements.txt
    ```sh
        fastapi
        uvicorn
        torch
        jinja2
        numpy
        pillow
        scikit-learn
        torch-geometric
        pandas
    ```
도커 이미지를 빌드할 때, 이 requirements.txt 파일에 명시된 모든 패키지들이 자동으로 설치됩니다.

-------

# To Do List
- 모델 정확도 향상
- 웹사이트 꾸미기
- 심장병 예측 이외의 다른 기능들 추가하기



## Reference
[1] https://www.kaggle.com/code/alexteboul/heart-disease-health-indicators-dataset-notebook  (kaggle dataset)

[2] https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset  (kaggle dataset)

[3] 오픈소스소프트웨어실습 week10 수업 자료

[4] https://pytorch.org/docs/stable/index.html pytorch documentation 

[5] https://scikit-learn.org/stable/user_guide.html scikit-learn user guide