import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('crows_pairs_anonymized.csv')

# 只保留 bias_type 为 'age' 的行
df_age = df[df['bias_type'] == 'age']

# 可选：将结果保存到新的 CSV 文件
df_age.to_csv('age.csv', index=False)