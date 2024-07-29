# 🌄 기상청 : 안개 시정구간 예측


## Abstract
> 안개의 시정거리와 시정구간에 대해 이해하고 안개의 시정구간을 예측한다.

<h2> 👪 Team </h2>

> Name : 해빛온담

<h3> 👪 Members </h3>
<table>
  <tr>
    <td> <div align=center> 👑 </div> </td>
    <td> <div align=center>  1 </div> </td>
    <td> <div align=center>  2 </div> </td>
  </tr>
  <tr>
    <td> <div align=center> <b>김다운</b> </div> </td>
    <td> <div align=center> <b>서상혁</b> </div> </td>
    <td> <div align=center> <b>신동혁</b> </div> </td>
  </tr>
  <tr>
    <td> <img alt="Github" src ="https://github.com/UpstageAILab/upstage-ml-regression-01/assets/76687996/0f945311-9828-4e50-a60c-fc4db3fa3b9d"  width="250"  height="300"/>  </td>
    <td> <img alt="Github" src ="https://github.com/UpstageAILab/upstage-ml-regression-01/assets/76687996/c4cb11ba-e02f-4776-97c8-9585ae4b9f1d"  width="250"  height="300"/>  </td>
    <td> <img alt="Github" src ="https://github.com/UpstageAILab/upstage-ml-regression-01/assets/76687996/a4dbcdb5-1d28-4b91-8555-1168abffc1d0"  width="250"  height="300"/>  </td>
  </tr>
  <tr>
    <td> <div align=center> <a href="https://github.com/Daw-ny"> <img alt="Github" src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/> </div> </td>
    <td>  <div  align=center>  <a href="https://github.com/Godjumo">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
    <td>  <div  align=center>  <a href="https://github.com/devhyuk96">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
  </tr>
</table>

<h3> 🛑 Role & Rule </h3>

> ### Ground Rule
> - 각자 연구하고 싶은 방향에 대해 연구하고 그 결과를 꼭 공유하도록 한다.
> - 본인이 세운 가설을 꼭 논리적으로 접근한 다음 확인이 되었을 때 다음 스텝으로 진행한다.
> - 최종적으로 앙상블 방법을 적용하기 위해 각 class의 확률을 계산해서 마무리하는 것으로 진행한다.
> - 22 ~ 24시에는 디스코드 접속


<table>
  <tr>
    <td> <div align=center> <b> 이름 </b> </div> </td>
    <td> <div align=center> <b> 역할 </b> </div> </td>
  </tr>
  <tr>
    <td> <div align=center> <b> 김다운 </b> </div> </td>
    <td> <b>EDA </b>(결측치 데이터 knnimputer처리, 이상치 탐색 및 재조정)</br> 
	 <b>모델링 진행 </b>(Catboost by cpu & gpu)</br>
	 <b>파생변수 생성 </b>(시간적 주기, 이슬점 온도, 안개 생성 조건)</br>
	 <b>대시보드 생성 </b>(Catboost를 모르는 사용자를 위한 대시보드 프로도타입 생성) </td>
  </tr>
  <tr>
    <td> <div align=center> <b> 서상혁 </b> </div> </td>
    <td> <b>EDA </b>(회귀 분석을 활용한 데이터 관계 설명, 결측치 보간법 정리 및 적용)</br>
	 <b>모델링 진행 </b>(Xgboost, LightGBM, AutoML, Oversampling/Undersampling 기법 적용)</br>
	 <b>파생변수 생성 </b>(log 또는 boxcox변환을 통한 관계 변화 적용)</br>
	 <b>대시보드 생성 </b>(연구자 또는 전문 지식을 가지고 있는 사람들에게 설명할 수 있는 프로도타입 생성) </td>
  </tr>
  <tr>
    <td> <div align=center> <b> 신동혁 </b> </div> </td>
    <td> <b>EDA </b>(결측값 대치, sliding window를 통한 사전 시점 학습)</br>
	 <b>모델링 진행 </b>(LSTM)</br>
	 <b>파생변수 생성 </b>(푸리에 변환을 통한 풍향 조정)</br>
	 <b>코드 및 데이터 정리 </b>(활용한 코드 및 데이터 정리 및 취합) </td>
  </tr>
</table>

<h3> 📽️ Project Intro </h3>

<table>
  <tr>
    <td> <div align=center> <b> Subject </b> </div> </td>
    <td> 불균형 클래스인 안개 시정구간 예측 </td>
  </tr>
  <tr>
    <td> <div align=center> <b> Processing </b> </div> </td>
    <td> 1. 평가 매트릭인 CSI score에 대해 이해하고 중요 포인트를 포커싱 </br>
  	 2. 도메인 지식과 연관지은 파생변수 생성 및 검증 </br>
     3. 각자 생각하는 모델을 활용하여 파생하여 만든 변수 공유하여 모델링 적용 </td>
  </tr>
  <tr>
    <td> <div align=center> <b> Develop Enviroment </b> </div> </td>
    <td> <tt>Tool</tt>: Jupyter Notebook, Python</td>
  </tr>
  <tr>
    <td> <div align=center> <b> Communication Enviroment </b> </div> </td>
    <td> <tt>Notion</tt>: 아이디어 브레인 스토밍, 프로젝트 관련 회의 내용 기록, 제출 모델 구현 방법 및 점수 기록 </br>
	 <tt>Discord</tt>: 실시간 비대면 회의 </td>
  </tr>
</table>

<h3> 📆 Project Procedure </h3>

>  자세한 진행 내용은 [notion](http://sixth-drum-9ac.notion.site)에서 확인하실 수 있습니다.

<h3> 📂 Project Structure </h3>

> - Code
>> 각 방법을 이름에 따른 방법으로 적용하면서 반복 실행하였습니다. 파일을 주로 모델링 방법에 따라서 나누었습니다.

<h3> ⚙️ Architecture </h3>
<table>
  <tr>
    <td> <div align=center> <b> 분류 </b> </div> </td>
    <td> <div align=center> <b> 내용 </b> </div> </td>
  </tr>
  <tr>
    <td> <div align=center> <b> 모델 </b> </div> </td>
    <td> <tt>Catboost</tt> </td>
  </tr>
  <tr>
    <td> <div align=center> <b> 데이터 </b> </div> </td>
    <td> 대회 자체 제공 데이터(용량이 너무 큰 관계로 업로드 불가)</td>
  </tr>
  <tr>
    <td> <div align=center> <b> 모델 평가 및 해석 </b> </div> </td>
    <td> CSI score을 활용하여 직접 계산, 비교 이후 최종적으로 score가 가장 큰 model을 test set 제출 </td>
  </tr>
</table>
