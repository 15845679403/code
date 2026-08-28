# 数据集说明
## 1.数据来源
滚动轴承振动数据集：https://engineering.case.edu/bearingdatacenter

包含4种运行工况：正常、内圈故障、外圈故障、滚动体故障。
原始mat文件存放于`data/mat_raw`。

## 2.目录说明
- `/data/mat_raw/`：原始mat数据集
- `/data/raw/`：mat转换后的csv振动文件，单列为vibration振动信号
- `/data/processed/`：预处理输出数据集

## 3.文件标签映射
|文件名|标签|工况说明|
|---|---|---|
|normal.csv|0|正常状态|
|98normal.csv|0|正常状态|
|inner.csv|1|内圈故障|
|106inner.csv|1|内圈故障|
|outer.csv|2|外圈故障|
|131out.csv|2|外圈故障|
|ball.csv|3|滚动体故障|
|119ball.csv|3|滚动体故障|

## 4.预处理流程
>工具脚本：`algorithm/all_data_tool.py`、`algorithm/data_preprocess.py`
1. mat转csv：读取mat中的DE_time/FE_time振动信号输出csv；
2. 读取csv振动信号；
3. 滑动平均滤波降噪；
4. 滑动窗口切分：窗口256，步长128；
5. 特征提取：
    - 时域特征：均值、方差、均方根、峰值、峭度、偏度
    - 频域特征：FFT幅值均值、方差、最大值
6. 全部样本拼接，分层按8:2划分训练集、测试集；
7. 基于训练集拟合StandardScaler标准化，保存scaler.pkl。

## 5.processed输出文件
- X_train.npy：训练集特征
- y_train.npy：训练集标签
- X_test.npy：测试集特征
- y_test.npy：测试集标签
- scaler.pkl：标准化缩放器，预测推理时加载使用
