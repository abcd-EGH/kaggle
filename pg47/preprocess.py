# 데이터프레임 복사
import pandas as pd

def derived_variables(df_input):
    df = df_input.copy()

    df = df.drop(columns=['id'])
    # 데이터 타입 변환
    df['Region_Code'] = df['Region_Code'].astype(int)
    df['Policy_Sales_Channel'] = df['Policy_Sales_Channel'].astype(int)

    # Vehicle_Damage 변환: Yes -> 1, No -> 0
    df['Vehicle_Damage'] = df['Vehicle_Damage'].apply(lambda x: 1 if x == 'Yes' else 0).astype(int)
    df['Previously_Insured'] = df['Previously_Insured'].astype(int)

    # 1. Insured_Vintage: Previously_Insured와 Vintage의 곱
    df['Insured_Vintage'] = df['Previously_Insured'] * df['Vintage']

    # 2. Region_Risk: 각 Region_Code 별 평균 Vehicle_Damage 비율
    region_risk = df.groupby('Region_Code')['Vehicle_Damage'].mean()
    df['Region_Risk'] = df['Region_Code'].map(region_risk)

    df = df.drop(columns=['Previously_Insured', 'Region_Code'])
    return df

def normalize_train_data(train):
    train_df = train.copy()
    # 수치형 데이터만 선택
    column_to_str = ['Driving_License', 'Vehicle_Damage','Response']

    for col in column_to_str:
        train_df[col] = train_df[col].astype(str)
    numeric_cols = train_df.select_dtypes(include=['int64', 'float64', 'int32']).columns
    
    # 평균과 표준편차를 저장할 딕셔너리
    stats = {}
    
    # 각 수치형 열에 대해 평균과 표준편차 계산 후 정규화 수행
    for col in numeric_cols:
        mean = train_df[col].mean()
        std = train_df[col].std()
        train_df[col] = (train_df[col] - mean) / std
        stats[col] = (mean, std)
    
    return train_df, stats

def normalize_test_data(test, stats):
    test_df = test.copy()
    # 수치형 데이터만 선택
    column_to_str = ['Driving_License', 'Vehicle_Damage', 'Response']

    for col in column_to_str:
        test_df[col] = test_df[col].astype(str)

    numeric_cols = test_df.select_dtypes(include=['int64', 'float64', 'int32']).columns
    
    # 평균과 표준편차를 사용하여 정규화
    for col in numeric_cols:
        mean, std = stats.get(col, (0, 1))  # 기본값으로 0, 1을 사용
        test_df[col] = (test_df[col] - mean) / std
    
    return test_df

def one_hot_encode(df_input, columns = ['Gender', 'Driving_License', 'Vehicle_Age', 'Vehicle_Damage']):
    df = df_input.copy()
    # 지정된 열의 데이터 타입을 문자열로 변환
    for col in columns:
        df[col] = df[col].astype(str)
    
    # Train 데이터에서 원핫 인코딩 수행
    df_encoded = pd.get_dummies(df, columns=columns, drop_first=True)
    
    return df_encoded

def convert_bool_to_numeric(df):
    # DataFrame 복사본을 생성하여 원본 데이터를 변경하지 않음
    df_converted = df.copy()
    
    # 모든 컬럼을 순회하며 bool 타입의 컬럼을 찾고, 해당 컬럼을 int 타입으로 변환
    for col in df_converted.columns:
        if df_converted[col].dtype == bool:
            df_converted[col] = df_converted[col].astype(int)
    
    return df_converted