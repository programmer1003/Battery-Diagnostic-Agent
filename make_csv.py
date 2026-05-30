import pandas as pd
import numpy as np

print("正在生成用于测试 LangGraph 和 TF-GDC 的电池数据...")

# 1. 生成 600 个正常的时间点数据 (电压 3.8V, 温度 25度)
time_steps = 600
voltage = np.random.normal(3.8, 0.05, time_steps)
current = np.random.normal(1.0, 0.1, time_steps)
temperature = np.random.normal(25.0, 1.0, time_steps)

# 2. 😈 故意制造致命异常！（在最后 20 个时间步：电压掉到 2.5V，温度飙升到 55度）
voltage[-20:] = np.random.normal(2.5, 0.1, 20)
temperature[-20:] = np.random.normal(55.0, 2.0, 20)

# 3. 组合成 DataFrame 并保存为 CSV
df = pd.DataFrame({
    'Time(s)': range(time_steps),
    'Voltage(V)': voltage,
    'Current(A)': current,
    'Temperature(C)': temperature
})

df.to_csv('mock_battery_fault_data.csv', index=False)
print("✅ 搞定！名为 'mock_battery_fault_data.csv' 的文件已生成在当前目录下！")