import pandas as pd
import numpy as np

# 生成 600 行完全健康的电池时序数据
# 电压稳定在 3.7V 左右，电流平稳，温度死死压在 25度 左右
data = {
    'Time': pd.date_range(start='2026-04-06 00:00:00', periods=600, freq='S'),
    'Voltage_V': np.random.normal(3.7, 0.01, 600),   # 极小波动的电压
    'Current_A': np.random.normal(10.0, 0.1, 600),   # 平稳放电
    'Temperature_C': np.random.normal(25.0, 0.5, 600) # 恒温
}

df = pd.DataFrame(data)
df.to_csv("healthy_battery_data.csv", index=False)
print("✅ 健康电池数据 (healthy_battery_data.csv) 已生成！")