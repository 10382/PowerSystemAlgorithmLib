import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

def xgboost_train(X, y, model_path="model/xgboost.pkl"):
    model = xgb.XGBRegressor(n_estimators=1000, objective ='reg:squarederror')
    # rfr = xgb.XGBRegressor(n_estimators=1000)
    model.fit(X.values, y)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

def xgboost_test(X, model=None, model_path="model/decompose/xgboost.pkl"):
    if model == None:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    return model.predict(X.values)

# 输入：model_name:需要对预测结果进行解释的模型, timestamp：需要解释的能耗时间戳, feature_dataset：数据集,feature_lists:通过shapley需要计算单独能耗的硬件
def shapley_hardware_power(model_name, timestamp, feature_dataset, feature_lists, feature_columns, model=None):
    # 预测的总能耗
    pre = xgboost_test(feature_dataset.iloc[timestamp:timestamp+1], model=model)
    # print(pre)
    zero_data = np.zeros((1,24))
    zero_dataframe = pd.DataFrame(zero_data, columns = feature_columns)
    C_power = pre - xgboost_test(zero_dataframe, model=model)
    # print("C_powr:",C_power)
    # 赋值需要计算的硬件特征值
    for feature_name in feature_lists:
        # print(zero_dataframe.iloc[timestamp:timestamp + 1])
        zero_dataframe.iloc[:,feature_name] = feature_dataset.iloc[timestamp:timestamp + 1, feature_name].values[0]
        # print(zero_dataframe.iloc[timestamp:timestamp + 1])
    # print("zero_dataframe",zero_dataframe)
    feature_power = xgboost_test(zero_dataframe, model=model) - C_power
    # print("F_powr:", feature_power)
    # 返回单独能耗
    return feature_power

# 输入：model:需要对预测结果进行解释的模型, timestamp：需要解释的能耗时间戳, feature_dataset：数据集
def shapley_idle_power(model_name, timestamp, feature_dataset, feature_columns, model=None):
    pre = xgboost_test(feature_dataset.iloc[timestamp:timestamp+1], model=model)
    # print("shapley_idle_power")
    zero_data = np.zeros((1,24))
    zero_dataframe = pd.DataFrame(zero_data, columns=feature_columns)
    C_power = pre - xgboost_test(zero_dataframe, model=model)
    return C_power

#输入：model_name:需要对预测结果进行解释的模型, timestamp：需要解释的能耗时间戳, feature_dataset：数据集,feature_lists:通过shapley需要计算单独能耗的硬件
def common_hardware_power(model_name, timestamp, feature_dataset, first_feature_list, second_feature_list,
                          feature_columns, model=None):
    # 预测的总能耗
    pre = xgboost_test(feature_dataset.iloc[timestamp:timestamp+1], model=model)
    # print("common_hardware_power(")
    zero_data = np.zeros((1,24))
    zero_dataframe = pd.DataFrame(zero_data, columns=feature_columns)
    first_hardware_power_shapley = shapley_hardware_power(model_name, timestamp = timestamp, feature_dataset = feature_dataset,
                                                          feature_lists=first_feature_list, feature_columns = feature_columns,
                                                          model=model)
    second_hardware_power_shapley = shapley_hardware_power(model_name, timestamp = timestamp, feature_dataset = feature_dataset,
                                                           feature_lists=second_feature_list, feature_columns = feature_columns,
                                                           model=model)
    for feature_name in first_feature_list:
        zero_dataframe.iloc[timestamp:timestamp + 1,feature_name] = feature_dataset.iloc[timestamp:timestamp + 1, feature_name].values[0]
    for feature_name in second_feature_list:
        zero_dataframe.iloc[timestamp:timestamp + 1,feature_name] = feature_dataset.iloc[timestamp:timestamp + 1, feature_name].values[0]
    feature_power = xgboost_test(zero_dataframe, model=model)
    common_hardware_power = first_hardware_power_shapley + second_hardware_power_shapley - feature_power
    return common_hardware_power


def feature_num(feature_dataset,feature_name):
    feature_list = list(feature_dataset)
    num_list = []
    for i in range(0,feature_dataset.shape[1]):
        if feature_name in feature_list[i]:
            num_list.append(i)
    return num_list

def hardware_power(feature_dataset, pre_power,timestamp, feature_columns, power_dict=None, model=None):
    i = timestamp
    C_value = shapley_idle_power("xgboost", i, feature_dataset, feature_columns = feature_columns, model=model)
    CPU_value = shapley_hardware_power("xgboost", i, feature_dataset, feature_num(feature_dataset,'cpu'), feature_columns = feature_columns, model=model)
    Disk_value = shapley_hardware_power("xgboost", i, feature_dataset, feature_num(feature_dataset,'disk'), feature_columns = feature_columns, model=model)
    Memory_value = shapley_hardware_power("xgboost", i, feature_dataset, feature_num(feature_dataset,'memory'), feature_columns = feature_columns, model=model)
    Net_value = shapley_hardware_power("xgboost", i, feature_dataset, feature_num(feature_dataset,'inter'), feature_columns = feature_columns, model=model)
    CPU_Disk_value = common_hardware_power("xgboost", i, feature_dataset, feature_num(feature_dataset,'cpu'),feature_num(feature_dataset,'disk'), feature_columns = feature_columns, model=model)
    Memory_CPU_value  = common_hardware_power("xgboost", i, feature_dataset, feature_num(feature_dataset,'cpu'),feature_num(feature_dataset,'memory'), feature_columns = feature_columns, model=model)
    CPU_Net_value = common_hardware_power("xgboost", i, feature_dataset, feature_num(feature_dataset,'cpu'),feature_num(feature_dataset,'inter'), feature_columns = feature_columns, model=model)
    Memory_Disk_value = common_hardware_power("xgboost", i, feature_dataset,feature_num(feature_dataset,'disk'),feature_num(feature_dataset,'memory'), feature_columns = feature_columns, model=model)
    Disk_Net_value = common_hardware_power("xgboost", i, feature_dataset, feature_num(feature_dataset,'disk'), feature_num(feature_dataset,'inter'), feature_columns = feature_columns, model=model)
    Memory_Net_value = common_hardware_power("xgboost", i, feature_dataset, feature_num(feature_dataset,'memory'),feature_num(feature_dataset,'inter'), feature_columns = feature_columns, model=model)
    # 能耗分解结果
    CPU_contribution = abs(CPU_value - Memory_CPU_value * (CPU_value / (CPU_value + Memory_value)) - CPU_Disk_value * (CPU_value / (CPU_value + Disk_value)) - CPU_Net_value * (CPU_value / (CPU_value + Net_value)))
    Disk_contribution = abs(Disk_value - CPU_Disk_value * (Disk_value / (CPU_value + Disk_value)) - Memory_Disk_value * (Disk_value / (Memory_value + Disk_value)) - Disk_Net_value * (Disk_value / (Disk_value + Net_value)))
    Memory_contribution = abs( Memory_value - Memory_CPU_value * (Memory_value / (CPU_value + Memory_value)) - Memory_Disk_value * (Memory_value / (Memory_value + Disk_value)) - Memory_Net_value * (Memory_value / (Memory_value + Net_value)))
    Net_contribution = abs(Net_value - CPU_Net_value * (Net_value / (CPU_value + Net_value)) - Disk_Net_value * (Net_value / (Disk_value + Net_value)) - Memory_Net_value * (Net_value / (Memory_value + Net_value)))
    power_specific = pre_power / (CPU_contribution+Disk_contribution+Memory_contribution+Net_contribution+C_value)
    CPU_power = CPU_contribution * power_specific
    Disk_power = Disk_contribution * power_specific
    Memory_power = Memory_contribution * power_specific
    Net_power = Net_contribution * power_specific
    C_power = C_value * power_specific
    if power_dict is None:
        power_list = {'other_power': C_power[0], 'CPU_power': CPU_power[0], 'Disk_power': Disk_power[0],'Memory_power': Memory_power[0], 'Net_power':Net_power[0]}
        return power_list
    else:
        power_dict['other_power'].append(C_power[0])
        power_dict['CPU_power'].append(CPU_power[0])
        power_dict['Disk_power'].append(Disk_power[0])
        power_dict['Memory_power'].append(Memory_power[0])
        power_dict['Net_power'].append(Net_power[0])
        return power_dict
