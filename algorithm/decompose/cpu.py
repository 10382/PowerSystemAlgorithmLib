import numpy as np

def decompose(vm_df_dict, cpu_metric, power_df, only_cpu=False, multi=False):
    cpu_usage_dict = dict()
    # 得到每个虚拟机的cpu使用率
    for vm_name in vm_df_dict.keys():
        cpu_usage_dict[vm_name] = vm_df_dict[vm_name].loc[:, cpu_metric].values
        # if multi:
        #     cpu_usage_dict[vm_name] = vm_df_dict[vm_name].loc[:, cpu_metric].values
        # else:
        #     cpu_usage_dict[vm_name] = vm_df_dict[vm_name].loc[:, cpu_metric].values[0]
    print(cpu_usage_dict)
    if only_cpu:
        return cpu_usage_dict
    cpu_total = np.sum(cpu_usage_dict.values())
    power = power_df.values[0]
    # cpu_total = sum(cpu_usage_dict.values())
    # power = power_df.values[0][0]
    # print(power)
    vm_power_dict = dict()
    for vm_name in cpu_usage_dict.keys():
        vm_power_dict[vm_name] = cpu_usage_dict[vm_name] / cpu_total * power
    return vm_power_dict


