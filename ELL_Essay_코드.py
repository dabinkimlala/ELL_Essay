###### 1. 데이터 구조 파악하기 #####

import numpy as np  
import pandas as pd  
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import RobertaTokenizer, RobertaModel
import torch
from torch.utils.data import DataLoader, Dataset


train = pd.read_csv("C:/Users/rlaek/OneDrive/바탕 화면/4-2/캡스톤\데이터/train.csv")
train.head(10)

test = pd.read_csv("C:/Users/rlaek/OneDrive/바탕 화면/4-2/캡스톤\데이터/test.csv")

missing_values = train.isnull().sum()
print(missing_values) # 결측치 없음. 

train.shape # 약 3900개의 에세이

##### 2. 각 변수별 분포 및 구조 파악 #####

train.describe()

for column in train.iloc[:, 2:].columns:
    print(f"Value counts for column '{column}':")
    print(train[column].value_counts())
    print("\n")

mean_values = train.mean(numeric_only=True)
median_values = train.median(numeric_only=True)

mean_values

median_values

train.hist(column=['cohesion','syntax','vocabulary','phraseology','grammar','conventions'], bins=20, figsize=(10, 10))
plt.xlim(1,5)
plt.show()

# 비율에 대한 그림 그리기 

bins = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
labels = ['1-1.5', '1.5-2', '2-2.5', '2.5-3', '3-3.5', '3.5-4', '4-4.5', '4.5-5']

for column in ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']:
    
    train[f'{column}_bin'] = pd.cut(train[column], bins=bins, labels=labels, include_lowest=True)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()  # 

for i, column in enumerate(['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']):
    distribution = train[f'{column}_bin'].value_counts(normalize=True).sort_index()
    distribution.plot(kind='bar', ax=axes[i], color='skyblue')
    axes[i].set_title(f'Distribution of {column}')
    axes[i].set_xlabel('Score Range')
    axes[i].set_ylabel('Proportion')
    axes[i].set_ylim(0, 0.5)  # 비율이므로 0에서 1 사이로 설정

plt.tight_layout()  
plt.show()

##### 3. 이상치 유무 확인 #####

valid_values = np.arange(1, 5.5, 0.5)

for column in ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']:
    invalid_rows = train[~train[column].isin(valid_values)]
    if len(invalid_rows) > 0:
        print(f"Column '{column}': 이상치가 있는 행들:")
        print(invalid_rows)
    else:
        print(f"Column '{column}': 이상치가 없습니다.")
        
#### 4. 변수간 상관관계, 연관성 파악 #####

correlation_matrix = train[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']].corr()
print(correlation_matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix of Essay Evaluation Metrics")
plt.show()

# Pairplot을 사용하여 변수 간 산점도 그리기
sns.pairplot(train[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']])
plt.suptitle("Scatterplot Matrix of Essay Evaluation Metrics", y=1.02)
plt.show()

# pairplot 크기 조정
sns.pairplot(train[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']],
             kind='kde', height=1.5)  # height 값 조정으로 크기 축소
plt.show()

##### 5. pca / clustering #####

pca = PCA(n_components=2)
pca_result = pca.fit_transform(train[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']])
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.title("PCA of Essay Evaluation Metrics")
plt.show()

explained_variance = pca.explained_variance_ratio_
print(explained_variance)

# PCA 결과에서 원점으로부터의 거리 계산
distances = np.sqrt(np.sum(pca_result**2, axis=1))

# 중앙에서 멀리 떨어진 데이터 포인트들 (예: 거리의 상위 10%)
threshold = np.percentile(distances, 95)  # 상위 5% 거리 계산
outliers = pca_result[distances > threshold]

# 전체 데이터 시각화 및 중앙에서 벗어난 데이터 시각화
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], label="All Data", alpha=0.5)
plt.scatter(outliers[:, 0], outliers[:, 1], color='red', label="Outliers (Far from Center)")
plt.title("PCA of Essay Evaluation Metrics with Outliers Highlighted")
plt.legend()
plt.show()

# 중앙에서 멀리 떨어진 데이터 인덱스 찾기
outlier_indices = np.where(distances > threshold)[0]
print(f"Outlier Indices: {outlier_indices}")

# 오른쪽으로 넓게 퍼진 데이터 필터링 (예: PCA 결과 x축이 특정 값 이상인 데이터)
right_outliers = pca_result[pca_result[:, 0] > 2]  # x축 값 기준으로 데이터 필터링
right_outlier_indices = np.where(pca_result[:, 0] > 2)[0]

# 원 데이터에서 오른쪽 데이터 추출
right_original_data = train.iloc[right_outlier_indices]

# 오른쪽 데이터 통계량 확인
print("오른쪽 데이터 통계량:")
print(right_original_data.describe())

# 오른쪽 데이터 분포 시각화
right_original_data.hist(bins=20, figsize=(10, 10))
plt.suptitle("Distribution of Right Outliers in Original Data", y=1.02)
plt.show()

# 왼쪽으로 넓게 퍼진 데이터 필터링 (예: PCA 결과 x축이 특정 값 이하인 데이터)
left_outliers = pca_result[pca_result[:, 0] < -2]  # x축 값 기준으로 데이터 필터링
left_outlier_indices = np.where(pca_result[:, 0] < -2)[0]

# 원 데이터에서 왼쪽 데이터 추출
left_original_data = train.iloc[left_outlier_indices]

# 왼쪽 데이터 통계량 확인
print("왼쪽 데이터 통계량:")
print(left_original_data.describe())

# 왼쪽 데이터 분포 시각화
left_original_data.hist(bins=20, figsize=(10, 10))
plt.suptitle("Distribution of Left Outliers in Original Data", y=1.02)
plt.show()

##### 6. 높은 점수와 낮은 점수에 대한 분석 #####

# 평가 지표 열들에 대한 점수의 총합 계산
train['total_score'] = train[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']].sum(axis=1)

# 상위 10개 데이터 추출
top_10 = train.nlargest(10, 'total_score')
print("Top 10 Essays with Highest Total Scores:")
print(top_10)

# 하위 10개 데이터 추출
bottom_10 = train.nsmallest(10, 'total_score')
print("\nBottom 10 Essays with Lowest Total Scores:")
print(bottom_10)

##### 7. 데이터 전처리, 텍스트 텍터화 #####
features = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar',  'conventions']
target = train[features]
target

text_train = train['full_text']
text_test = test['full_text']

# text_train과 text_test를 합치기 위해 concat 사용
text = pd.concat([text_train, text_test], ignore_index=True)
text

# 동일한 전처리를 위해 데이터 합침.

count_words = text.str.findall(r'(\w+)').str.len()
print(count_words.sum())

# 텍스트에서 단어 수를 세는 작업을 수행하는 코드

import re

""" Cleaning Text """
text = text.str.lower() # 모든 텍스트를 소문자로 변환

text = text.apply(lambda x : re.sub("[^a-z]\s","",x) ) # 특수 문자 및 숫자 제거

text = text.str.replace("#", "") # # 기호를 제거

text = text.apply(lambda x: ' '.join([w for w in x.split() if len(w)>2 and len(w)<8])) # 3자 미만, 8자 이상의 단어는 제거.
# 3자 미만의 단어는 I / a와 같이 의미가 중요하지 않은 단어가 많이 포함되어있고 8자 이상의 단어는 오타일 가능성이나 너무 특정한 의미를 갖고 있을 경우가 많음 .
# removing stopwords

count_words = text.str.findall(r'(\w+)').str.len()
print(count_words.sum())

most_freq_words = pd.Series(' '.join(text).lower().split()).value_counts()[:25]
text = text.apply(lambda x : " ".join(word for word in x.split() if word not in most_freq_words ))
print(most_freq_words)

count_words = text.str.findall(r'(\w+)').str.len()
print(count_words.sum())

#텍스트 데이터에서 가장 빈번하게 나타나는 단어 25개를 찾아서 제거한 후, 남은 단어들로 단어 수를 다시 계산하는 작업을 수행. 이 과정은 분석에서 빈번하게 반복되는 단어들(예: 불용어)이 결과에 미치는 영향을 줄이고, 의미 있는 단어들만 남기기 위한 것임.

apostrophe_dict = {
"ain't": "am not / are not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is",
"i'd": "I had / I would",
"i'd've": "I would have",
"i'll": "I shall / I will",
"i'll've": "I shall have / I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}
apostrophe_dict

def lookup_dict(txt, dictionary):
    for word in txt.split():
        if word.lower() in dictionary:
            if word.lower() in txt.split():
                txt = txt.replace(word, dictionary[word.lower()])
    return txt

text = text.apply(lambda x: lookup_dict(x,apostrophe_dict))

from collections import Counter
from itertools import chain

v = text.str.split().tolist()
c = Counter(chain.from_iterable(v))
text = [' '.join([j for j in i if c[j] > 1]) for i in v]
text = pd.Series(text)

total_word = 0
for x,word in enumerate(text):
    num_word = len(word.split())
    #print(num_word)
    total_word = total_word + num_word
print(total_word)

y = target
X = text[: len(train)]
X_test = text[len(train) :]
X.shape, X_test.shape, y.shape

vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=0.01)

X_test = np.array(X_test).tolist()
X_test = list(map(''.join, X_test))

X_tfIdf = vectorizer_tfidf.fit_transform(X)
X_test_tfIdf = vectorizer_tfidf.transform(X_test)
print(vectorizer_tfidf.get_feature_names_out()[:5])

###### 8. 모델 돌리기 #####
# 8-1 svm 
chain = MultiOutputRegressor(SVR())
chain.fit(X_tfIdf, y)
print(chain.score(X_tfIdf,y))

# 모델 학습
chain = MultiOutputRegressor(SVR())
chain.fit(X_tfIdf, y)

# R² 점수 출력
r2_score = chain.score(X_tfIdf, y)
print(f"R² Score: {r2_score}")

# 예측값 계산
y_pred = chain.predict(X_tfIdf)

# MSE 계산
mse_score = mean_squared_error(y, y_pred, multioutput='raw_values')  # 각 타겟 변수의 MSE
average_mse = mean_squared_error(y, y_pred)  # 전체 평균 MSE

# MSE 출력
print(f"MSE for each target: {mse_score}")
print(f"Average MSE: {average_mse}")

# test 데이터에 대한 예측 수행
y_test_pred = chain.predict(X_test_tfIdf)

# 예측 결과 출력
print("Predicted scores for the test data:\n", y_test_pred)

# 8-2 xgboost

# 다중 출력 XGBoost 회귀 모델
xgb_model = MultiOutputRegressor(XGBRegressor(objective='reg:squarederror')) # 선형회귀모형을 띄고 있기에 reg:squarederror으로 설정
xgb_model.fit(X_tfIdf, target)
print("XGBoost Model Score:", xgb_model.score(X_tfIdf, target))

# 예측값 계산
y_pred = xgb_model.predict(X_tfIdf)

# MSE 계산
mse = mean_squared_error(target, y_pred)
print("Mean Squared Error (MSE):", mse)

# test 데이터에 대한 예측 수행
y_test_pred = xgb_model.predict(X_test_tfIdf)

# 예측 결과 출력
print("Predicted scores for the test data:\n", y_test_pred)

# 8-3 랜덤포레스트 

# 랜덤 포레스트 회귀 모델 생성
rf_model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
rf_model.fit(X_tfIdf, target)

# 훈련 데이터에 대한 예측
y_pred_train = rf_model.predict(X_tfIdf)

# 모델 평가 - 훈련 데이터
from sklearn.metrics import r2_score

train_mse = mean_squared_error(target, y_pred_train)
train_r2_score = r2_score(target, y_pred_train)

print("Training MSE:", train_mse)
print("Training R² Score:", train_r2)

# test 데이터에 대한 예측
y_test_pred = rf_model.predict(X_test_tfIdf)

# 예측 결과 출력
print("Predicted scores for the test data:\n", y_test_pred)

##### 9. Train 데이터의 적합성 검증을 위한 실험 #####

# A. 상위권 점수 데이터 제거 실험

# 1) 높은 점수의 에세이 기준 설정
# 평균 점수가 4.5 이상인 데이터를 필터링 (높은 점수의 에세이)
features = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
high_score_essays = train[train[features].mean(axis=1) >= 4]

# 나머지 데이터를 학습 데이터로 사용
train_filtered = train[train[features].mean(axis=1) < 4]

# 2) 텍스트와 타겟 변수 추출
# 학습 데이터
X_train = train_filtered['full_text']
y_train = train_filtered[features]

# 테스트 데이터 (제거된 높은 점수 에세이)
X_test_high = high_score_essays['full_text']
y_test_high = high_score_essays[features]

# 3) TF-IDF 벡터화
# 학습 데이터와 테스트 데이터를 병합하여 TF-IDF 피팅
text_combined = pd.concat([X_train, X_test_high], ignore_index=True)
vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=0.01)

X_combined_tfidf = vectorizer_tfidf.fit_transform(text_combined)

# 학습 데이터와 테스트 데이터 분리
X_train_tfidf = X_combined_tfidf[:len(X_train)]
X_test_high_tfidf = X_combined_tfidf[len(X_train):]

# 4) 모델 학습
# MultiOutputRegressor와 SVR 사용
chain = MultiOutputRegressor(SVR())
chain.fit(X_train_tfidf, y_train)

# 5) 모델 평가
# 학습 데이터 성능 평가
y_train_pred = chain.predict(X_train_tfidf)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = chain.score(X_train_tfidf, y_train)

print(f"Training MSE: {train_mse}")
print(f"Training R² Score: {train_r2}")

# 테스트 데이터 성능 평가 (높은 점수 에세이)
y_test_high_pred = chain.predict(X_test_high_tfidf)
test_high_mse = mean_squared_error(y_test_high, y_test_high_pred)
test_high_r2 = chain.score(X_test_high_tfidf, y_test_high)

print(f"Test High Scores MSE: {test_high_mse}")
print(f"Test High Scores R² Score: {test_high_r2}")

# 6) 결과 비교
# 실제 점수와 예측 점수 비교 출력
print("Actual High Scores:\n", y_test_high)
print("Predicted High Scores:\n", y_test_high_pred)

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(y_test_high.values.flatten(), label="Actual Scores", marker='o')
plt.plot(y_test_high_pred.flatten(), label="Predicted Scores", marker='x')
plt.title("Actual vs Predicted Scores for High Scoring Essays")
plt.xlabel("Samples")
plt.ylabel("Scores")
plt.legend()
plt.show()

# 4) 모델 학습
# MultiOutputRegressor와 RandomForestRegressor 사용
chain = MultiOutputRegressor(RandomForestRegressor(random_state=42))
chain.fit(X_train_tfidf, y_train)

# 5) 모델 평가
# 학습 데이터 성능 평가
y_train_pred = chain.predict(X_train_tfidf)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = chain.score(X_train_tfidf, y_train)

print(f"Training MSE: {train_mse}")
print(f"Training R² Score: {train_r2}")

# 테스트 데이터 성능 평가 (높은 점수 에세이)
y_test_high_pred = chain.predict(X_test_high_tfidf)
test_high_mse = mean_squared_error(y_test_high, y_test_high_pred)
test_high_r2 = chain.score(X_test_high_tfidf, y_test_high)

print(f"Test High Scores MSE: {test_high_mse}")
print(f"Test High Scores R² Score: {test_high_r2}")

# 6) 결과 비교
# 실제 점수와 예측 점수 비교 출력
print("Actual High Scores:\n", y_test_high)
print("Predicted High Scores:\n", y_test_high_pred)

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(y_test_high.values.flatten(), label="Actual Scores", marker='o')
plt.plot(y_test_high_pred.flatten(), label="Predicted Scores", marker='x')
plt.title("Actual vs Predicted Scores for High Scoring Essays")
plt.xlabel("Samples")
plt.ylabel("Scores")
plt.legend()
plt.show()

# 4) 모델 학습
# MultiOutputRegressor와 XGBRegressor 사용
chain = MultiOutputRegressor(XGBRegressor(random_state=42))
chain.fit(X_train_tfidf, y_train)

# 5) 모델 평가
# 학습 데이터 성능 평가
y_train_pred = chain.predict(X_train_tfidf)
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = chain.score(X_train_tfidf, y_train)

print(f"Training MSE: {train_mse}")
print(f"Training R² Score: {train_r2}")

# 테스트 데이터 성능 평가 (높은 점수 에세이)
y_test_high_pred = chain.predict(X_test_high_tfidf)
test_high_mse = mean_squared_error(y_test_high, y_test_high_pred)
test_high_r2 = chain.score(X_test_high_tfidf, y_test_high)

print(f"Test High Scores MSE: {test_high_mse}")
print(f"Test High Scores R² Score: {test_high_r2}")

# 6) 결과 비교
# 실제 점수와 예측 점수 비교 출력
print("Actual High Scores:\n", y_test_high)
print("Predicted High Scores:\n", y_test_high_pred)

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(y_test_high.values.flatten(), label="Actual Scores", marker='o')
plt.plot(y_test_high_pred.flatten(), label="Predicted Scores", marker='x')
plt.title("Actual vs Predicted Scores for High Scoring Essays")
plt.xlabel("Samples")
plt.ylabel("Scores")
plt.legend()
plt.show()

##### B. 상위권 데이터에 오류를 추가한 민감도 실험 #####

# 데이터 불러오기
modified_high_score_essays = pd.read_excel("C:/Users/rlaek/OneDrive/바탕 화면/4-2/캡스톤/높은 점수 에세이 오류로 수정.xlsx")  # 수정된 만점 에세이 데이터

# 학습 데이터 준비
text_column = 'full_text'
features = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
X_train = train[text_column]
y_train = train[features]

# 수정된 만점 데이터 준비
X_test = modified_high_score_essays[text_column].astype(str)  # 수정된 만점 에세이를 test 데이터로 사용

# TF-IDF 벡터화
vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=0.01)
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

# 모델 학습
model = MultiOutputRegressor(SVR())
model.fit(X_train_tfidf, y_train)

# 수정된 만점 에세이에 대한 예측
y_test_pred = model.predict(X_test_tfidf)

# 예측 결과 출력
print("Predicted Scores for Modified High-Scoring Essays:")
print(y_test_pred)

# 시각화
plt.figure(figsize=(10, 6))
x = np.arange(len(y_test_pred))
for i, feature in enumerate(features):
    plt.plot(x, y_test_pred[:, i], label=f"Predicted {feature}", marker='o', alpha=0.7)

plt.xticks(x, [f"Essay {i+1}" for i in range(len(X_test))])
plt.title("Predicted Scores for Modified High-Scoring Essays")
plt.xlabel("Essays")
plt.ylabel("Scores")
plt.legend()
plt.show()

# 모델 학습 (랜덤 포레스트)
rf_model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
rf_model.fit(X_train_tfidf, y_train)

# 수정된 만점 에세이에 대한 예측
y_test_pred_rf = rf_model.predict(X_test_tfidf)

# 예측 결과 출력
print("Predicted Scores for Modified High-Scoring Essays (Random Forest):")
print(y_test_pred_rf)

# 시각화
plt.figure(figsize=(10, 6))
x = np.arange(len(y_test_pred_rf))
for i, feature in enumerate(features):
    plt.plot(x, y_test_pred_rf[:, i], label=f"Predicted {feature}", marker='o', alpha=0.7)

plt.xticks(x, [f"Essay {i+1}" for i in range(len(X_test))])
plt.title("Predicted Scores for Modified High-Scoring Essays (Random Forest)")
plt.xlabel("Essays")
plt.ylabel("Scores")
plt.legend()
plt.show()

# TF-IDF 벡터화
vectorizer_tfidf = TfidfVectorizer(stop_words='english', max_df=0.5, min_df=0.01)
X_train_tfidf = vectorizer_tfidf.fit_transform(X_train)
X_test_tfidf = vectorizer_tfidf.transform(X_test)

# 모델 학습 (XGBoost)
xgb_model = MultiOutputRegressor(XGBRegressor(random_state=42))
xgb_model.fit(X_train_tfidf, y_train)

# 수정된 만점 에세이에 대한 예측
y_test_pred_xgb = xgb_model.predict(X_test_tfidf)

# 예측 결과 출력
print("Predicted Scores for Modified High-Scoring Essays (XGBoost):")
print(y_test_pred_xgb)

# 시각화
plt.figure(figsize=(10, 6))
x = np.arange(len(y_test_pred_xgb))
for i, feature in enumerate(features):
    plt.plot(x, y_test_pred_xgb[:, i], label=f"Predicted {feature}", marker='o', alpha=0.7)

plt.xticks(x, [f"Essay {i+1}" for i in range(len(X_test))])
plt.title("Predicted Scores for Modified High-Scoring Essays (XGBoost)")
plt.xlabel("Essays")
plt.ylabel("Scores")
plt.legend()
plt.show()

# Roberta 딥러닝 

# Hugging Face의 transformers와 PyTorch 관련 라이브러리

# 데이터 파일 불러오기

train_df = pd.read_csv('/content/train.csv')  # 학습 데이터
test_df = pd.read_csv('/content/test.csv')    # 테스트 데이터
sample_sub = pd.read_csv('/content/sample_submission.csv')  # 제출 파일 형식

# 특수 문자 제거 및 텍스트 정리
train_df['full_text'] = train_df['full_text'].str.replace(r'[\n\r\t]', ' ', regex=True)
test_df['full_text'] = test_df['full_text'].str.replace(r'[\n\r\t]', ' ', regex=True)

class TextDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        target = self.targets[index]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'target': torch.tensor(target, dtype=torch.float)
        }

class RoBERTaRegressor(torch.nn.Module):
    def __init__(self, pretrained_model_name):
        super(RoBERTaRegressor, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        self.regressor = torch.nn.Linear(self.roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # 각 토큰의 출력 벡터의 평균 계산
        return self.regressor(pooled_output)  # 회귀값 반환

# 파라미터 설정
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")  # RoBERTa 토크나이저
max_len = 128  # 입력 텍스트의 최대 길이
batch_size = 16  # 배치 크기

# 학습 및 테스트 데이터
train_texts = train_df['full_text'].tolist()
test_texts = test_df['full_text'].tolist()

# 6개의 타겟 변수
targets = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
train_targets = [train_df[target].values for target in targets]

# 결과 저장
predictions = []

for i, target in enumerate(train_targets):
    print(f"Training model for {targets[i]}...")

    # 데이터셋 및 데이터로더 생성
    train_dataset = TextDataset(train_texts, target, tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 모델 초기화
    model = RoBERTaRegressor("roberta-base")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)  # Adam 옵티마이저
    criterion = torch.nn.MSELoss()  # 평균 제곱 오차 손실 함수

    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 학습 루프
    for epoch in range(3):  # 에포크 수는 3으로 설정
        epoch_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(-1), targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} - Loss: {epoch_loss / len(train_loader)}")

    # 테스트 데이터 예측
    model.eval()
    test_dataset = TextDataset(test_texts, [0] * len(test_texts), tokenizer, max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            test_preds.extend(outputs.squeeze(-1).cpu().numpy())

    predictions.append(test_preds)

test_df

predictions

# 타겟 변수 
target = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

# 결과를 데이터프레임으로 변환
result_df = pd.DataFrame({"text_id": test_df["text_id"]})  # text_id 추가
for i, target in enumerate(target):
    result_df[target] = predictions[i]  # 각 타겟 변수의 예측값 추가

print(result_df)

##### RoBERTa 딥러닝 #####

# Hugging Face의 transformers와 PyTorch 관련 라이브러리

# 데이터 파일 불러오기
train = pd.read_csv('C:/Users/rlaek/OneDrive/바탕 화면/4-2/캡스톤\데이터/train.csv')  # 학습 데이터
test = pd.read_csv('C:/Users/rlaek/OneDrive/바탕 화면/4-2/캡스톤\데이터/test.csv')    # 테스트 데이터
sample_sub = pd.read_csv('/content/sample_submission.csv')  # 제출 파일 형식

# 특수 문자 제거 및 텍스트 정리
train['full_text'] = train['full_text'].str.replace(r'[\n\r\t]', ' ', regex=True)
test['full_text'] = test['full_text'].str.replace(r'[\n\r\t]', ' ', regex=True)

class TextDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        target = self.targets[index]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'target': torch.tensor(target, dtype=torch.float)
        }

class RoBERTaRegressor(torch.nn.Module):
    def __init__(self, pretrained_model_name):
        super(RoBERTaRegressor, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        self.regressor = torch.nn.Linear(self.roberta.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)  # 각 토큰의 출력 벡터의 평균 계산
        return self.regressor(pooled_output)  # 회귀값 반환

# 파라미터 설정
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")  # RoBERTa 토크나이저
max_len = 128  # 입력 텍스트의 최대 길이
batch_size = 16  # 배치 크기

# 학습 및 테스트 데이터
train_texts = train['full_text'].tolist()
test_texts = test['full_text'].tolist()

# 6개의 타겟 변수
features = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
train_targets = [train[feature].values for feature in features]

# 결과 저장
predictions = []

for i, target in enumerate(train_targets):
    print(f"Training model for {features[i]}...")

    # 데이터셋 및 데이터로더 생성
    train_dataset = TextDataset(train_texts, target, tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 모델 초기화
    model = RoBERTaRegressor("roberta-base")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)  # Adam 옵티마이저
    criterion = torch.nn.MSELoss()  # 평균 제곱 오차 손실 함수

    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 학습 루프
    for epoch in range(3):  # 에포크 수는 3으로 설정
        epoch_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.squeeze(-1), targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} - Loss: {epoch_loss / len(train_loader)}")

    # 테스트 데이터 예측
    model.eval()
    test_dataset = TextDataset(test_texts, [0] * len(test_texts), tokenizer, max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask)
            test_preds.extend(outputs.squeeze(-1).cpu().numpy())

    predictions.append(test_preds)

test

predictions

# 타겟 변수 
features = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

# 결과를 데이터프레임으로 변환
result_df = pd.DataFrame({"text_id": test["text_id"]})  # text_id 추가
for i, feature in enumerate(features):
    result_df[feature] = predictions[i]  # 각 타겟 변수의 예측값 추가

print(result_df)
