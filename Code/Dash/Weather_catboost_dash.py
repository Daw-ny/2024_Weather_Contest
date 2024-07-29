##########################################################################
# Import packages

# dash
from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx
import dash_bootstrap_components as dbc

# basic
import pandas as pd
import numpy as np
import pickle
import shap

# plotly
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

# model
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')

##########################################################################
# get abs shap
def get_ABS_SHAP(df_shap, df):
    
    # ground
    cri = [
        df['ground'] == 'A',
        df['ground'] == 'B',
        df['ground'] == 'C',
        df['ground'] == 'D'
    ]
    con = [
        0, 1, 2, 3
    ]
    df['ground'] = np.select(cri, con, default = 4)

    # Make a copy of the input data
    shap_v = pd.DataFrame(df_shap)
    feature_list = df.columns
    shap_v.columns = feature_list
    df_v = df.copy().reset_index().drop('index',axis=1)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i], df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list), pd.Series(corr_list)], axis=1).fillna(0)
 
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')
    
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2 = k2.sort_values(by='SHAP_abs',ascending = True)
    
    k2_f = k2[['Variable', 'SHAP_abs', 'Corr']]
    k2_f['SHAP_abs'] = k2_f['SHAP_abs'] * np.sign(k2_f['Corr'])
    k2_f.drop(columns='Corr', inplace=True)
    k2_f.rename(columns={'SHAP_abs': 'SHAP'}, inplace=True)
    
    return k2_f

# shap to plotly
def shap_summary_plot_to_plotly(shap_values, X):
    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_df_mean = shap_df.abs().mean().sort_values(ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=shap_df_mean.values,
        y=shap_df_mean.index,
        orientation='h'
    ))
    
    # Edit the layout
    fig.update_layout(
                        title='SHAP Summary Plot',
                        xaxis_title='Mean |SHAP Value|',
                        yaxis_title='Feature',
                        title_x = 0.5,
                        title_xanchor = 'center',
                        title_font_color = 'black',
                        title_font_family = 'NanumSquare',
                        template='simple_white')

    # 색상조절
    fig.update_traces(# marker_color = 히스토그램 색, 
                        # marker_line_width = 히스토그램 테두리 두깨,                            
                        # marker_line_color = 히스토그램 테두리 색,
                        marker_opacity = 0.4,
                        )
    
    return fig
##########################################################################
# import path
path_dir = './Data/'

# import data
train = pd.read_csv(path_dir + 'train_final_preprocess.csv')
test = pd.read_csv(path_dir + 'test_final_preprocess.csv')

# SHAP values
with open(path_dir + 'shap_values.pickle', 'rb') as f:
    shap_values = pickle.load(f)

# load model
cb = CatBoostClassifier()
cb.load_model('./model/cb_best.model')

##########################################################################
# need dictionary
# 계절
season = ['All', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# 변수 종류
value = ['hm', 'sun10', 'ta', 'ts', 'ws10_deg', 'ws10_ms', 'month', 'time', 're', 'ground', 'year']
value_k = ['상대습도', '10분 평균 일사량', '기온', '지면온도', '10분 평균 풍향', '10분 평균 풍속', '월', '시간', '강수유무', '지형 종류']

# 변수 변환 역변환
dic_r_val = {'상대습도':'hm', '강수유무': 're', '10분 평균 일사량': 'sun10', '기온': 'ta', '지면온도': 'ts', 
             '10분 평균 풍향': 'ws10_deg', '10분 평균 풍속': 'ws10_ms', '지형 종류': 'ground'}

# class
class_names = ['고밀도 안개', '중밀도 안개', '저밀도 안개', '안개 없음']

# use label
use_label_x = ['hm', 're', 'sun10', 'ta', 'ts',
                'ws10_ms', 'ground', 'dew_point',
                'sin_time', 'cos_time', 'sin_month',
                'cos_month', 'diff_air-dew', 'diff_ts-dew',
                'fog_risk', 'sin_deg', 'cos_deg']

# test_x
test_x = test[use_label_x]

##########################################################################
# data handeling 

# fog
# 이전 class
train['class_before'] = train['class'].shift(1)
train['class_before'][train['class_before'].isna()] = train['class'][train['class_before'].isna()]

# 안개 발생 변화 시점 찾기
cri = [
    (train['class'] == 4) & (train['class_before'] != 4),
    (train['class'] != 4) & (train['class_before'] == 4)
]
con = [
    -1, 1
]
train['fog'] = np.select(cri, con, default = 0)

# 안개발생 시작 시점
fog = train[train['fog'] == 1].reset_index(drop = True)

# 안개 발생 종료시점을 추가
fog['EndDateTime'] = train[train['fog'] == -1]['DateTime'].reset_index(drop = True)

# 날짜 데이터로 바꾸기
fog['DateTime'] = pd.to_datetime(fog['DateTime'])
fog['EndDateTime'] = pd.to_datetime(fog['EndDateTime'])

# 안개 지속시간 구하기
fog['last_fog'] = (fog['EndDateTime'] - fog['DateTime']).dt.seconds/3600

##########################################################################
# abs for shap by 4classes
foo_all = pd.DataFrame()
for k, v in list(enumerate(class_names)):

    foo = get_ABS_SHAP(shap_values[k], test_x)
    foo['class'] = v
    foo_all = pd.concat([foo_all, foo])

fig_shap = px.bar(foo_all[foo_all['Variable'] != 'ground'],
                    x='SHAP',
                    y='Variable',
                    color='class')

# Edit the layout
fig_shap.update_layout(
                    title='SHAP Summary Plot',
                    title_x = 0.5,
                    title_xanchor = 'center',
                    title_font_color = 'black',
                    title_font_family = 'NanumSquare',
                    template='simple_white')

# 색상조절
fig_shap.update_traces(# marker_color = 히스토그램 색, 
                    # marker_line_width = 히스토그램 테두리 두깨,                            
                    # marker_line_color = 히스토그램 테두리 색,
                    marker_opacity = 0.4,
                    )

##########################################################################
# Main Dashboard
app = Dash(external_stylesheets=[dbc.themes.MORPH], suppress_callback_exceptions=True)

# App layout
app.layout = dbc.Container([

    # 대시보드 제목
    html.Br(), # 띄어쓰기
    dbc.Row([
        html.H1(children='Weighted Catboost를 활용한 안개 시정 구간 예측')
    ]),
    
    dbc.Row([
        html.Hr() # 구분선
    ]),

    dbc.Row([
        html.Div(id='page-container'),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Pagination(
                id='pagenation',
                max_value=3,
                previous_next=True,
                active_page=1,  # 기본 페이지를 1로 설정
            )
        ], width={"size": 6, "offset": 3})
    ],
        style={
            'position': 'fixed',
            'bottom': '0',
            'left': '0',
            'width': '100%',
            'padding': '10px 0',  # 상하 여백 추가
            'box-shadow': '0 -2px 5px rgba(0, 0, 0, 0.1)'  # 상단 그림자 추가
        }
    ),

])

##########################################################################
# Page 1 components: 대시보드를 보여주기 전 각종 정보 및 주의사항에 대해 알려주는 페이지
first_comp = [
    dbc.Row([
                dbc.Alert("Dash 사용 설명서 및 주의사항", color="primary"),
            ]),

    dbc.Row([
        
        dcc.Markdown("""
                    ## **제작 의도**

                     - 해당 대시보드는 **분석한 결과를 한눈에 볼 수 있도록 시각화**하고 안개 정보와 예측이 필요한 사용자를 위해 제작되었습니다.  
                     - 배경 지식이 없는 일반인을 대상으로 적용하였으며 최대한 해석에 용이하게 보여주기 위해 제작하였습니다.  
                     - 대회에서 사용된 모델로 내가 원하는 시점의 정보를 입력하여 안개 발생 가능성이 있는지 추정하기 위해 직접 대입할 수 있도록 **3페이지**에 마련하였습니다.                       

                    ## **How to Use**

                     - 페이지는 현재 페이지를 포함하여 총 3페이지로 이루어져 있습니다.  
                     - 2페이지는 분석에 사용한 3년치의 안개 발생 정보와 각 연도별 기본정보, 최종 선정된 Weight Catboost에서 사용한 모델의 변수 중요도를 보여주고 있습니다.  
                     - 각 라디오 버튼과 드롭다운을 통해 선택에 따른 결과를 출력할 수 있습니다.  
                     - 3페이지는 직접적으로 값을 대입하여 안개 발생 구간이 어디에 포함되어 있는지 확인할 수 있습니다.  
                     - 텍스트 칸으로 되어있는 값은 숫자로 입력하며 드롭다운은 해당되는 옵션을 선택하여 **'Analyze'** 버튼을 누릅니다.  
                     - 왼쪽 아래에 예측하는데 계산된 파생 변수를 출력해주며 오른쪽에는 해당하는 안개의 정도를 표시합니다.  
                     안개는 다음과 같이 **4단계**로 이루어져 있습니다.  

                     > 1: 고밀도 안개  
                     > 2: 중밀도 안개  
                     > 3: 저밀도 안개  
                     > 4: 안개 없음  

                     ## **주의할 점**
                     - 모든 기능은 로딩을 기다려야 정상 작동 합니다. 만일 2페이지의 그래프가 에러로 인해 표시되지 않는다면 새로고침을 통해 페이지를 재실행 시키시길 바랍니다.  
                     - 2페이지의 오른쪽 변수 중요도의 x축의 숫자는 상대적인 수치를 의미합니다. 따라서 0을 기준으로 방향과 정도를 통해 **절대적인 해석을 하면 안됩니다.**  
                       ex) 상대습도가 -2, 풍속이 -1일 경우 상대 습도가 풍속에 2배의 영향을 끼친다. (잘못된 해석)  
                     - 3페이지에서 각 변수의 입력 범위는 다음과 같습니다. 이 범위를 벗어나는 값을 입력할 경우 에러를 표시할 수 있습니다.  

                     > 상대습도[%] : 0 ~ 100  
                     > 풍향[deq] : 0 ~ 360  
                     > 풍속[m/s] : 0 ~  
                     > 일사량[MJ] : 0 ~  
                     > 지면온도[℃], 기온[℃] : 제한 없음
                     
                     """)

    ]),
    
    # 아래 컨텐츠 가리는 것 방지
    dbc.Row([
        html.Br() # 구분선
    ]),

    dbc.Row([
        html.Br() # 구분선
    ]),

    dbc.Row([
        html.Br() # 구분선
    ]),

    dbc.Row([
        html.Br() # 구분선
    ]),

    dbc.Row([
        html.Br() # 구분선
    ]),
]

# page 2 components: 기본적인 변수 정보와 모델링에서의 변수 중요도를 알 수 있는 페이지, 인터렉티브한 그래프도 여기에 포함
second_comp = [

    dbc.Row([
        
        # 1. 안개 발생이 얼마나 자주 일어나는지에 대해 정리
        dbc.Col([

            dbc.Row([
                dbc.Alert("안개 발생 횟수", color="primary"),
            ]), 
            
            ###################
            # radio button
            dbc.Row([
                dbc.Label("지역 단위"),
            ]),
            
            dbc.Row([
                dbc.RadioItems(
                    options=[
                        {"label": "구분 없음", "value": 1},
                        {"label": "구분 있음", "value": 2},
                    ],
                    value=1,
                    id="radio-ground",
                    inline=True,
                ),
            ]),
            
            dbc.Row([
                html.Br() # 구분선
            ]),

            dbc.Row([
                dbc.Label("안개 지속시간"),
            ]),
            
            dbc.Row([
                dbc.RadioItems(
                    options=[
                        {"label": "전체", "value": 1},
                        {"label": "1시간 이상", "value": 2},
                        {"label": "1시간 미만", "value": 3},
                    ],
                    value=1,
                    id="radio-duration_time",
                    inline=True,
                ),
            ]),
            ###################
            # 그래프1 : 시간별 안개 발생량 
            dbc.Row([
                dcc.Graph(figure={}, id='time_histogram')
            ]),

            dbc.Row([
                html.Hr() # 구분선
            ]),
            
            # 월 선택 드롭다운
            dbc.Row([
                dbc.Label("월 선택"),
            ]),
            
            dbc.Row([
                dbc.Select(season,
                    value = 'All',
                    id = 'drop-month', size="sm")
            ]),

            dbc.Row([
                html.Br() # 구분선
            ]),

            # 그래프2 : 월별 안개 발생량
            dbc.Row([
                dcc.Graph(figure={}, id='month_histogram')
            ]),

            # 아래 컨텐츠 가리는 것 방지
            dbc.Row([
                html.Br() # 구분선
            ]),

            dbc.Row([
                html.Br() # 구분선
            ]),

            dbc.Row([
                html.Br() # 구분선
            ]),

            dbc.Row([
                html.Br() # 구분선
            ]),

            dbc.Row([
                html.Br() # 구분선
            ]),
        ]),
        
        ##########################################################################
        # 2. 현재 사용되고 있는 데이터의 분포에 대해 확인할 수 있도록 제시 단 train을 보여준다는 가정
        dbc.Col([

            dbc.Row([
                dbc.Alert("분석에 사용된 요인의 연도별 분포", color="primary"),
            ]),
            
            # 월 선택 드롭다운
            dbc.Row([
                dbc.Label("변수 선택"),
            ]),
            
            dbc.Row([
                dbc.Select(['상대습도', '10분 평균 일사량', '기온', '지면온도', '10분 평균 풍향', '10분 평균 풍속', '강수유무'],
                    value = '상대습도',
                    id = 'drop-valuename', size = 'sm')
            ]),

            dbc.Row([
                html.Br() # 구분선
            ]),
            
            dbc.Row([
                dbc.Label("연도"),
            ]),
            
            dbc.Row([
                dbc.RadioItems(
                    options=[
                        {"label": "I", "value": 1},
                        {"label": "J", "value": 2},
                        {"label": "K", "value": 3},
                        {"label": "L", "value": 4},
                    ],
                    value=1,
                    id="radio-year",
                    inline=True,
                ),
            ]),

            # 그래프
            dbc.Row([
                dcc.Graph(figure={}, id='single_boxplot')
            ]),

            dbc.Row([
                html.Br() # 구분선
            ]),

            dbc.Row([
                dcc.Graph(figure={}, id='single_histogram')
            ]),

        ]),

        ##########################################################################
        # 3. feature importances by SHAP
        dbc.Col([

            dbc.Row([
                dbc.Alert("Weigthed Catboost로 인한 Feature Importance", color="primary"),
            ]),
            
            dbc.Row([
                dbc.Label("분석 결과의 전체 변수 중요도"),
            ]),

            # 그래프
            dbc.Row([
                dcc.Graph(figure = fig_shap)
            ]),
            
            dbc.Row([
                html.Br() # 구분선
            ]),

            dbc.Row([
                dbc.Label("class 선택"),
            ]),

            # 드롭다운
            dbc.Row([
                dbc.Select(['안개 없음', '저밀도 안개', '중밀도 안개', '고밀도 안개'],
                    value = '안개 없음',
                    id = 'drop-classname', size='sm')
            ]),

            dbc.Row([
                html.Br() # 구분선
            ]),

            # 그래프
            dbc.Row([
                dcc.Graph(figure={}, id='shap_select_fig')
            ]),

        ]),

    ]),
]

# page 3 components: 값을 입력하면 그 값을 예측할 수 있도록 해주는 페이지
third_comp = [
    dbc.Row([
        dbc.Alert(" 내가 가진 데이터로 입력하여 모델이 추정하는 안개의 정도 확인하기 ", color="primary"),
    ]), 

    dbc.Row([
        html.Hr() # 구분선
    ]),

    dbc.Row([
        dbc.Label("대입값 입력"),
    ]),

    # 입력값 받기
    dbc.Row([
        dbc.Col([
            dbc.Row([
                html.Label('상대습도'),
            ]),
            dbc.Row([
                dbc.Input(value = 50, type='number', min = 0.0001, max = 100, id='hm_values', size="md"), 
            ]),
        ], style={'display': 'inline-block', 'margin-right': '20px',}),

        dbc.Col([
            dbc.Row([
                html.Label('10분 평균 일사량'),
            ]),
            dbc.Row([
                dbc.Input(value = 0.053, type='number', min = 0, id='sun10_values', size="md"), 
            ]),
        ], style={'display': 'inline-block', 'margin-right': '20px',}),

        dbc.Col([
            dbc.Row([
                html.Label('기온'),
            ]),
            dbc.Row([
                dbc.Input(value = 25, type='number', id='ta_values', size="md"), 
            ]),
        ], style={'display': 'inline-block', 'margin-right': '20px',}),

        dbc.Col([
            dbc.Row([
                html.Label('지면온도'),
            ]),
            dbc.Row([
                dbc.Input(value = 30, type='number', id='ts_values', size="md"), 
            ]),
        ], style={'display': 'inline-block', 'margin-right': '20px',}),

        dbc.Col([
            dbc.Row([
                html.Label('10분 평균 풍향'),
            ]),
            dbc.Row([
                dbc.Input(value = 92, type='number', min = 0, max = 360, id='ws10_deg_values', size="md"), 
            ]),
        ], style={'display': 'inline-block', 'margin-right': '20px',}),

    ], style={'margin-right': '20px'}),

    dbc.Row([
        html.Br() # 구분선
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Row([
                html.Label('10분 평균 풍속'),
            ]),
            
            dbc.Row([
                dbc.Input(value = 1.2, type='number', min = 0, id='ws10_ms_values', size="md"), 
            ]),
        ], style={'display': 'inline-block', 'margin-right': '20px'}),

        dbc.Col([
            dbc.Row([
                html.Label('월'),
            ]),
            
            dbc.Row([
                dbc.Select([i for i in range(1, 13)], value = 5, id='month_values', size="md"), 
            ]),
        ], style={'display': 'inline-block', 'margin-right': '20px'}),

        dbc.Col([
            dbc.Row([
                html.Label('시간'),
            ]),
            
            dbc.Row([
                dbc.Select([i for i in range(24)], value = 13, id='time_values', size="md"), 
            ]),
        ], style={'display': 'inline-block', 'margin-right': '20px'}),

        dbc.Col([
            dbc.Row([
                html.Label('강수유무'),
            ]),
            
            dbc.Row([
                dbc.Select(['No', 'Yes'], value = 'No', id='re_values', size="md"), 
            ]),
        ], style={'display': 'inline-block', 'margin-right': '20px'}),

        dbc.Col([
            dbc.Row([
                html.Label('지형 종류'),
            ]),
            
            dbc.Row([
                dbc.Select(['내륙', '산간', '동해', '서해', '남해'], value = '내륙', id='ground_values', size="md"), 
            ]),
        ], style={'display': 'inline-block', 'margin-right': '20px'}),

    ], style={'margin-right': '20px'}),

    dbc.Row([
        html.Br() # 구분선
    ]),

    # 제출 버튼
    dbc.Row([
        dbc.Button("Analyze", color="success", className="me-md-1", id='analysis', n_clicks=0),
    ], className="d-grid gap-2 d-md-flex justify-content-md-end", ),

    dbc.Row([
        html.Br() # 구분선
    ]),

    dbc.Row([
        html.Hr() # 구분선
    ]),
    
    # 분석 결과 문구 나타내주기
    dbc.Row([
        # 1. 분석에 사용할 파생변수 계산하여 값 가져오기
        dbc.Col([

            dbc.Row([
                dcc.Markdown("""
                        ## **기존 변수와 파생 변수의 목록 및 계산값**
                """)
            ]),

            dbc.Row([
                
                ### 해당 파라미터 도표를 넣자
                html.Div(id = 'table-use_values')
            
            ]),
        
        ]),

        # 2. Class 결과
        dbc.Col([
            dbc.Row([
                dcc.Markdown("""
                        #### **해당 입력값으로 추정한 클래스는 다음과 같습니다.**
                """)
            ]),

            dbc.Row([
                html.Div(id = 'class_type'),
            ]),
        ]),
    ]),

    # 아래 컨텐츠 가리는 것 방지
    dbc.Row([
        html.Br() # 구분선
    ]),

    dbc.Row([
        html.Br() # 구분선
    ]),

    dbc.Row([
        html.Br() # 구분선
    ]),

    dbc.Row([
        html.Br() # 구분선
    ]),

    dbc.Row([
        html.Br() # 구분선
    ]),
]


##########################################################################
# Dynamic Dashboard functions

### 페이지 번호별로 다른 요인 표시하기
@app.callback(
    Output('page-container', 'children'),
    Input('pagenation', 'active_page')
)
def display_page(page):
    if page == 1:
        return first_comp
    
    elif page == 2:
        return second_comp
    
    elif page == 3:
        return third_comp
    
    return []


# 시간단위 히스토그램
@callback(
    Output(component_id='time_histogram', component_property='figure'),
    Input(component_id='radio-ground', component_property='value'),
    Input(component_id='radio-duration_time', component_property='value')
)
def time_varing_hist(ground, duration):

    # 지역
    if ground == 2:
        color = '지형종류'
        barmode = 'group'

    else:
        color = None
        barmode = None

    # 시간 단위
    if duration == 2:
        dt = fog[fog['last_fog'] >= 1]

    elif duration == 3:
        dt = fog[fog['last_fog'] < 1]

    else:
        dt = fog

    # 이름 바꾸기
    cri = [
        dt['ground'] == 'A',
        dt['ground'] == 'B',
        dt['ground'] == 'C',
        dt['ground'] == 'D',
    ]
    con = [
        '내륙', '산간', '동해', '서해'
    ]
    dt['지형종류'] = np.select(cri, con, default = '남해')

    # Figure  생성
    # 1시간 이상만 모아서
    fig = px.histogram(dt,
                    x = "hour",
                    color=color,
                    barmode=barmode)

    fig.update_layout(title_text="시간대별 안개 발생 histogram",
                        title_x = 0.5,
                        title_xanchor = 'center',
                        title_font_color = 'black',
                        title_font_family = 'NanumSquare',
                        template='simple_white')

    fig.update_traces(# marker_color = 히스토그램 색, 
                        # marker_line_width = 히스토그램 테두리 두깨,                            
                        # marker_line_color = 히스토그램 테두리 색,
                        marker_opacity = 0.4,
                        )

    return fig


# 월 단위 히스토그램
@callback(
    Output(component_id='month_histogram', component_property='figure'),
    Input(component_id='radio-ground', component_property='value'),
    Input(component_id='radio-duration_time', component_property='value'),
    Input(component_id='drop-month', component_property='value')
)
def hour_varing_hist(ground, duration, months):

    # 지역
    if ground == 2:
        color = '지형종류'
        barmode = 'group'

    else:
        color = None
        barmode = None

    # 시간 단위
    if duration == 2:
        data = fog[fog['last_fog'] >= 1]

    elif duration == 3:
        data = fog[fog['last_fog'] < 1]

    else:
        data = fog

    # 선택 월
    dic_month = {'All': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                 'Jan': [1], 'Feb': [2], 'Mar': [3], 'Apr': [4], 'May': [5], 'Jun': [6], 'Jul': [7], 'Aug': [8], 'Sep': [9], 'Oct': [10], 'Nov':[11], 'Dec':[12]}

    # 이름 바꾸기
    cri = [
        data['ground'] == 'A',
        data['ground'] == 'B',
        data['ground'] == 'C',
        data['ground'] == 'D',
    ]
    con = [
        '내륙', '산간', '동해', '서해'
    ]
    data['지형종류'] = np.select(cri, con, default = '남해')

    # Figure  생성
    # 1시간 이상만 모아서
    fig = px.histogram(data[data['month'].isin(dic_month[months])],
                    x = "month",
                    color=color,
                    barmode=barmode)

    fig.update_layout(title_text="계절별 안개 발생 histogram",
                        title_x = 0.5,
                        title_xanchor = 'center',
                        title_font_color = 'black',
                        title_font_family = 'NanumSquare',
                        template='simple_white')

    fig.update_traces(# marker_color = 히스토그램 색, 
                        # marker_line_width = 히스토그램 테두리 두깨,                            
                        # marker_line_color = 히스토그램 테두리 색,
                        marker_opacity = 0.4,
                        )

    return fig

# 선택한 변수 boxplot
@callback(
    Output(component_id='single_boxplot', component_property='figure'),
    Input(component_id='drop-valuename', component_property='value'),
    Input(component_id='radio-year', component_property='value')
)
def single_box(selected_value, select_years):
    
    # selected years
    if select_years == 4:
        data = test

    else:
        data = train

    # 선택 연도
    dic_year = {
                    1: 'I',
                    2: 'J',
                    3: 'K',
                    4: 'L'
                }

    # Figure  생성
    fig = px.box(data[data['year'] == dic_year[select_years]],
                y = dic_r_val[selected_value]
                )
    
    # Edit the layout
    fig.update_layout(title_text=f"{dic_year[select_years]}년도 {selected_value}의 boxplot",
                        title_x = 0.5,
                        title_xanchor = 'center',
                        title_font_color = 'black',
                        title_font_family = 'NanumSquare',
                        template='simple_white')

    # 색상조절
    fig.update_traces(# marker_color = 히스토그램 색, 
                        # marker_line_width = 히스토그램 테두리 두깨,                            
                        # marker_line_color = 히스토그램 테두리 색,
                        marker_opacity = 0.4,
                        )

    return fig

# boxplot and histogram for one variable
@callback(
    Output(component_id='single_histogram', component_property='figure'),
    Input(component_id='drop-valuename', component_property='value'),
    Input(component_id='radio-year', component_property='value')
)
def single_hist(selected_value, select_years):

    # selected years
    if select_years == 4:
        data = test

    else:
        data = train

    # 선택 연도
    dic_year = {
                    1: 'I',
                    2: 'J',
                    3: 'K',
                    4: 'L'
                }

    # Figure  생성
    fig = px.histogram(data[data['year'] == dic_year[select_years]],
                x = dic_r_val[selected_value]
                )
    
    # Edit the layout
    fig.update_layout(title_text=f"{dic_year[select_years]}년도 {selected_value}의 histogram",
                        title_x = 0.5,
                        title_xanchor = 'center',
                        title_font_color = 'black',
                        title_font_family = 'NanumSquare',
                        template='simple_white')

    # 색상조절
    fig.update_traces(# marker_color = 히스토그램 색, 
                        # marker_line_width = 히스토그램 테두리 두깨,                            
                        # marker_line_color = 히스토그램 테두리 색,
                        marker_opacity = 0.4,
                        )
    return fig

# by class SHAP
@callback(
    Output(component_id='shap_select_fig', component_property='figure'),
    Input(component_id='drop-classname', component_property='value')
)
def shap_total(colname):

    # 데이터 대입하기
    test_x = test[use_label_x]

    # dic
    dic_class = {
        '안개 없음': 3,
        '저밀도 안개': 2,
        '중밀도 안개': 1,
        '고밀도 안개': 0
    }

    # 선택 그래프
    fig2 = shap_summary_plot_to_plotly(shap_values[dic_class[colname]], test_x)

    return fig2


# k-means 결과 분석
@callback(
    Output('table-use_values', 'children'),
    Output('class_type', 'children'),
    Input('analysis', 'n_clicks'),
    [Input(component_id=f'{value[i]}_values', component_property='value') for i in range(10)]
)
def result_k_means(clk, hm, sun10, ta, ts, ws10_deg, ws10_ms, month, time, re, ground):

    if "analysis" == ctx.triggered_id:
        # value = ['hm', 're', 'sun10', 'ta', 'ts', 'ws10_deg', 'ws10_ms', 'ground', 'month', 'time', 'year']
        # value_k = ['상대습도', '강수유무', '10분 평균 일사량', '기온', '지면온도', '10분 평균 풍향', '10분 평균 풍속', '지형 종류', '월', '시간']

        # use label
        # use_label_x = ['hm', 're', 'sun10', 'ta', 'ts',
        #                 'ws10_ms', 'ground', 'dew_point',
        #                 'sin_time', 'cos_time', 'sin_month',
        #                 'cos_month', 'diff_air-dew', 'diff_ts-dew',
        #                 'fog_risk', 'sin_deg', 'cos_deg']
        
        # dic for ground
        dic_ground = {
            '내륙': 'A', '산간': 'B', '동해': 'C', '서해':'D', '남해': 'E'
        }

        # dataframe for input value
        tmp = pd.DataFrame({
                'hm': [hm],
                're': [0 if re == 'No' else 1], 
                'sun10': [sun10], 
                'ta': [ta], 
                'ts': [ts], 
                'ws10_deg': [ws10_deg], 
                'ws10_ms': [ws10_ms], 
                'ground': [dic_ground[ground]],  
                'month': [int(month)],  
                'time': [int(time)]
            })

        # 파생변수 계산
        # 시간 사이클 변수
        tmp['sin_time'] = np.sin(2 * np.pi * tmp['time'] / 24)
        tmp['cos_time'] = np.cos(2 * np.pi * tmp['time'] / 24)

        # 계절 사이클 변수 - 월별 주기
        tmp['sin_month'] = np.sin(2 * np.pi * tmp['month'] / 12)
        tmp['cos_month'] = np.cos(2 * np.pi * tmp['month'] / 12)


        # Magnus 공식 상수
        a = 17.27
        b = 237.7

        # 알파 값 계산
        tmp['alpha'] = (a * tmp['ta']) / (b + tmp['ta']) + np.log(tmp['hm'] / 100.0)

        # 이슬점온도 계산
        tmp['dew_point'] = (b * tmp['alpha']) / (a - tmp['alpha'])

        # 온도조건 미리 계산하기
        tmp['diff_air-dew'] = tmp['ta'] - tmp['dew_point']
        tmp['diff_ts-dew'] = tmp['ts'] - tmp['dew_point']


        # AWS 좌표
        cri = [
            # 5단계: high risk
            (tmp['hm'] >= 97) & (tmp['ws10_ms'] <= 7) & (tmp['re'] == 0),

            # 4단계: middle risk
            (tmp['hm'] < 97) & (tmp['hm'] >= 95) & (tmp['ws10_ms'] <= 7) & (tmp['re'] == 0),

            # 3단계: Low risk
            (tmp['hm'] < 95) & (tmp['hm'] >= 90) & (tmp['ws10_ms'] <= 7) & (tmp['re'] == 0),

            # 2단계: Risk not estimates
            (tmp['hm'] >= 90)
        ]

        con = [
            4, 3, 2, 1
        ]

        tmp['fog_risk'] = np.select(cri, con, default = 0)

        # 풍향
        tmp['sin_deg'] = np.sin(tmp['ws10_deg'] * np.pi / 180)
        tmp['cos_deg'] = np.cos(tmp['ws10_deg'] * np.pi / 180)

        # 정리
        sample = tmp[use_label_x]
        #################################################
        # 예측값 뽑기
        pred_proba2 = cb.predict_proba(sample)

        # 예측값 가져오기
        cnum = np.argmax(pred_proba2, axis = 1)

        #################################################
        # 표 만들기
        # header
        table_header = [
            html.Thead(html.Tr([html.Th("사용 변수"), html.Th("값")]))
        ]
        
        # varnames
        varname = ['상대습도', '강수유무', '10분 평균 일사량', '기온', '지면온도', '10분 평균 풍속', '지형 종류', '이슬점 온도', '시간의 sin 변환',
         '시간의 cos 변환', '계절의 sin 변환', '계절의 cos 변환', '기온과 이슬점 온도의 차이', '지면온도와 이슬점 온도의 차이', 'AWS 안개위험지수',
         '풍향 sin 방향', '풍향 cos 방향']

        # body
        body = []
        for n, s in zip(varname, sample.columns):
            body.append(html.Tr([html.Td(n), html.Td(sample[s])]))

        # body 채우기
        table_body = [html.Tbody(body)]

        # 도표 
        table = html.Div([dbc.Table(table_header + table_body, bordered=True, color="success")])

        #################################################
        # class 분류하기

        color = ['firebrick', 'Gold', 'DarkGreen', 'royalblue']
        chtml = html.Div(class_names[cnum[0]], style={'color': color[cnum[0]], 'font-size': '100px', 'font-weight': 'bold', 'text-align': 'center'})

        return table, chtml
    
    else:
        return html.Div([dbc.Table(html.Thead(html.Tr([html.Th("사용 변수"), html.Th("값")])), color="success", bordered=True)]), ''
    
##########################################################################
# play
if __name__ == '__main__':
    app.run(debug=True)