# oss_personal_project_phase_1
2021314455 최연제

# 지원 Operating Systems

Docker를 이용하여 실행하는 프로젝트이므로, 다음 운영 체제에서 지원됩니다:

| OS      | 지원 여부 |
|---------|-----------|
| Windows | ⭕         |
| Linux   | ⭕         |
| MacOS   | ⭕         |


# **인공지능 기반 심장병 예측 모델 웹사이트**

심장병에는 여러 유형이 있지만, 대부분의 사람들은 흉통, 심장마비, 또는 갑작스러운 심장 정지와 같은 증상이 나타난 후에야 자신이 이 질병을 가지고 있다는 것을 알게 됩니다. 심장마비와 같은 부정적인 결과가 발생하기 전에 심장 질환을 정확하게 예측할 수 있는 예방 조치가 필요합니다. 고혈압, 고혈중 콜레스테롤, 흡연이 심장 질환의 세 가지 주요 위험 요소인데, 이를 포함한 7가지 수치를 사용하여 심장병 위험을 감지할 수 있습니다.

#### About the dataset:
Behavioral Risk Factor Surveillance System (BRFSS)는 미국 질병통제예방센터(CDC)가 매년 실시하는 건강 관련 전화 설문 조사입니다. 가져온 데이터셋은 심장병의 이진 분류를 위해 사용될 253,680개의 정리된 BRFSS 2015 설문 응답을 사용합니다. 원본 데이터셋의 330개의 특징 중, 7개만을 사용했습니다.

**사용한 features**:
1. High Blood Pressure: Have you EVER been told by a doctor, nurse or other health professional that your blood pressure is high?
2. High Cholesterol: Have you EVER been told by a doctor, nurse or other health professional that your blood cholesterol is high?
3. BMI 지수 (Body Mass Index)
4. Smoking: Have you smoked at least 100 cigarettes in your entire life?
5. Fruit Consumption: Do you consume fruit 1 or more times per day?
6. Vegetable Consumption: Do you consume vegetables 1 or more times per day?
7. Alcohol Consumption: Are you a heavy drinker?


*Accuracy: 0.9032910100673116*

### 심장병 예측 모델 웹사이트 실행 방법
이 프로젝트는 Docker를 사용하여 운영 체제에 관계없이 일관된 환경에서 실행할 수 있습니다. 아래는 Windows, Linux, MacOS에서 웹사이트를 실행하는 방법입니다.

##### Windows, Linux, MacOS 공통 실행 방법
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


- Docker를 사용하여 실행되므로, Docker가 설치되어 있으면 Windows, Linux, MacOS에서 동일한 방법으로 실행할 수 있습니다.

- 이미 사용 중인 포트가 있는 경우, `-p <host_port>:8000` 형식으로 다른 포트를 지정하여 실행할 수 있습니다.
    ```sh
    docker run -it --name webserver -p 8080:8000 disease_prediction
    ```
  그런 다음 [http://localhost:8080/](http://localhost:8080/)에 접속합니다.


### 실행 예시
![alt text](image.png)


### Reference
[1] https://www.kaggle.com/code/alexteboul/heart-disease-health-indicators-dataset-notebook  (kaggle dataset)
[2] https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset  (kaggle dataset)

