'''import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.linalg import Vectors
import joblib
def get_spark():
    return SparkSession.builder \
        .appName("StreamlitSparkModel") \
        .getOrCreate()

def UI(spark):
    option = st.selectbox(
    "Chọn model",
    ("Linear Regression", "XGboost", "Random Forest"),
    )
    st.title("GPA Predictor")
    st.header("This is an app to predict your next semester GPA based on the previous semester result")
    num_semesters = st.selectbox("Số học kỳ đã học:", options=[3, 4, 5, 6, 7])
    model_path = ""
    if option == "Linear Regression":
        linear_model_path = f"/spark_model/linear_logistic_{num_semesters}"
        model = LinearRegressionModel.load(linear_model_path)
    elif option == "Random Forest":
        randomforst_model_path = f"/spark_model/random_forest_{num_semesters}"
        model = RandomForestRegressor.load(randomforst_model_path)
# Tạo danh sách lưu điểm học kỳ
    grades = []

# Hiển thị input tương ứng với số học kỳ
    for i in range(1, num_semesters + 1):
        grade = st.number_input(f"Điểm trung bình học kỳ {i}:", min_value=0.0, max_value=10.0, step=0.1, key=f"hk{i}")
        grades.append(grade)

# Các thông tin khác nếu cần thêm (năm sinh, giới tính, v.v.)
    namsinh = st.number_input("Năm sinh:", min_value=1980, max_value=2020, value=2000)
    gioitinh = st.selectbox("Giới tính:", options=["Nam", "Nữ"])
    gioitinh_val = 1 if gioitinh == "Nam" else 0

# Nút dự đoán (sẽ dùng model ở đây)
    if st.button("Dự đoán điểm học kỳ tiếp theo"):
        features = [namsinh, gioitinh_val] + grades
        vector = Vectors.dense(features)

    # Tạo DataFrame Spark từ vector
        df = spark.createDataFrame([(vector,)], ["features"])

        prediction = model.transform(df).collect()[0]["prediction"]

        st.success(f"Điểm dự đoán học kỳ sau: {prediction:.2f}")
if __name__ == "__main__":
    spark = get_spark()
    UI(spark)'''

import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.linalg import Vectors
import pandas as pd
import joblib
def get_spark():
    return SparkSession.builder \
        .appName("StreamlitSparkModel") \
        .getOrCreate()

def UI(spark):
    st.title("GPA Predictor")
    st.header("This is an app to predict your next semester GPA based on the previous semester result")
    option = st.selectbox(
        "Chọn model",
        ("Linear Regression", "XGboost", "Random Forest"),
    )

    num_semesters = st.selectbox("Số học kỳ đã học:", options=[3, 4, 5, 6, 7])

    # Hiển thị input tương ứng với số học kỳ
    grades = []
    for i in range(1, num_semesters + 1):
        grade = st.number_input(f"Điểm trung bình học kỳ {i}:", min_value=0.0, max_value=10.0, step=0.1, key=f"hk{i}")
        grades.append(grade)

    namsinh = st.number_input("Năm sinh:", min_value=1980, max_value=2020, value=2000)
    gioitinh = st.selectbox("Giới tính:", options=["Nam", "Nữ"])
    gioitinh_val = 1 if gioitinh == "Nam" else 0

    if st.button("Dự đoán điểm học kỳ tiếp theo"):
        features = [namsinh, gioitinh_val] + grades

        if option == "Linear Regression":
            model_path = f"/spark_model/linear_logistic_{num_semesters}"
            model = LinearRegressionModel.load(model_path)
            vector = Vectors.dense(features)
            df = spark.createDataFrame([(vector,)], ["features"])
            prediction = model.transform(df).collect()[0]["prediction"]
        
        elif option == "Random Forest":
            model_path = f"/spark_model/random_forest_{num_semesters}"
            model = RandomForestRegressionModel.load(model_path)
            vector = Vectors.dense(features)
            df = spark.createDataFrame([(vector,)], ["features"])
            prediction = model.transform(df).collect()[0]["prediction"]

        elif option == "XGboost":
            model_path = f"../saved_model/xgboost_{num_semesters}.pkl"
            model = joblib.load(model_path)
            input_df = pd.DataFrame([features], columns=["namsinh", "gioitinh"] + [f"dtbhk_{i}" for i in range(1, num_semesters + 1)])
            prediction = model.predict(input_df)[0]

        st.success(f"Điểm dự đoán học kỳ sau: {prediction:.2f}")

if __name__ == "__main__":
    spark = get_spark()
    UI(spark)

