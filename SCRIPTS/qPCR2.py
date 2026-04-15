import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# ================= 1. 设置参数区 =================
file_paths = [
    'EXP1.xlsx',
    'EXP2.xlsx',
    'EXP4.xlsx'
]
reference_gene = 'RPS7'
control_sample = 'A23T'
sd_threshold = 0.5

# ================= 2. 数据汇总读取与清理 =================
df_list = []
for file in file_paths:
    try:
        temp_df = pd.read_excel(file)
        temp_df['Batch'] = file
        df_list.append(temp_df)
    except FileNotFoundError:
        print(f"⚠️ 警告：找不到文件 {file}，请检查文件名！")

df = pd.concat(df_list, ignore_index=True)
df = df.dropna(subset=['Target', 'Sample'])
df['Cq'] = pd.to_numeric(df['Cq'], errors='coerce')
df = df.dropna(subset=['Cq'])

# ================= 3. 自动检测并剔除异常复孔 =================
print("-" * 50)
print("开始执行跨批次数据质控 (自动剔除各批次 SD > 0.5 的离群孔)...")

outlier_indices = []

for (batch, sample, target), g in df.groupby(['Batch', 'Sample', 'Target']):
    if len(g) >= 3 and g['Cq'].std() > sd_threshold:
        mean_val = g['Cq'].mean()
        outlier_idx = (g['Cq'] - mean_val).abs().idxmax()
        outlier_indices.append(outlier_idx)

        bad_cq = g.loc[outlier_idx, 'Cq']
        bad_well = g.loc[outlier_idx, 'Well'] if 'Well' in g.columns else '未知孔位'

        print(f"  [剔除] {batch} | {sample} - {target} 组: 孔位 {bad_well} (Cq={bad_cq:.2f})")

if outlier_indices:
    df = df.drop(index=outlier_indices)

print("质控完成，开始汇总计算...")
print("-" * 50)

# ================= 4. qPCR 数据汇总计算 =================
ref_mean = df[df['Target'] == reference_gene].groupby(['Batch', 'Sample'])['Cq'].mean().reset_index()
ref_mean = ref_mean.rename(columns={'Cq': 'Ref_Cq_Mean'})

df = pd.merge(df, ref_mean, on=['Batch', 'Sample'])
df['Delta_Cq'] = df['Cq'] - df['Ref_Cq_Mean']

control_mean = df[df['Sample'] == control_sample].groupby(['Batch', 'Target'])['Delta_Cq'].mean().reset_index()
control_mean = control_mean.rename(columns={'Delta_Cq': 'Control_Delta_Cq_Mean'})

df = pd.merge(df, control_mean, on=['Batch', 'Target'])
df['Delta_Delta_Cq'] = df['Delta_Cq'] - df['Control_Delta_Cq_Mean']
df['Fold_Change'] = 2 ** (-df['Delta_Delta_Cq'])

plot_data = df[df['Target'] != reference_gene]

# ================= 5. T检验计算 =================
p_values = {}
targets = plot_data['Target'].unique()

for target in targets:
    group_control = plot_data[(plot_data['Target'] == target) & (plot_data['Sample'] == control_sample)][
        'Delta_Cq'].dropna().values
    group_treat = plot_data[(plot_data['Target'] == target) & (plot_data['Sample'] != control_sample)][
        'Delta_Cq'].dropna().values

    if len(group_control) > 1 and len(group_treat) > 1:
        t_stat, p_val = ttest_ind(group_control, group_treat, equal_var=False)
        p_values[target] = p_val
    else:
        p_values[target] = np.nan

# ================= 6. 数据可视化 (按基因自动批量出图) =================
# 【核心修改 1】：A23T设为黑色，A23设为科研蓝(royalblue)
custom_colors = {'A23T': 'black', 'A23': 'royalblue'}

for target in targets:
    target_data = plot_data[plot_data['Target'] == target]
    if target_data.empty:
        continue

    # 画布调整得稍微瘦高一点，比例更协调
    fig, ax = plt.subplots(figsize=(5, 7))

    # 【核心修改 2】：X轴设为 'Sample'，这样柱子下方就会自动印上 A23T 和 A23
    sns.barplot(
        data=target_data,
        x='Sample',
        y='Fold_Change',
        hue='Sample',
        order=['A23T', 'A23'],
        hue_order=['A23T', 'A23'],
        palette=custom_colors,
        edgecolor='black',
        linewidth=1.5,
        ax=ax,
        capsize=0.1,
        err_kws={'linewidth': 1.5, 'color': 'black'},
        zorder=3
    )

    plt.axhline(1, color='black', linestyle='--', linewidth=1.2, zorder=4)
    ax.grid(axis='y', linestyle=':', alpha=0.6, zorder=0)

    max_y = target_data['Fold_Change'].max()
    ax.set_ylim(0, max_y * 1.35)

    p_val = p_values.get(target, np.nan)
    if not pd.isna(p_val):
        # 【核心修改 3】：将星号替换为具体的 P 值
        if p_val < 0.001:
            p_text = 'p < 0.001'
        else:
            p_text = f'p = {p_val:.3f}'  # 保留3位小数

        line_y = max_y + 0.15
        text_y = line_y + 0.02

        # 因为 X 轴变成了 Sample，所以两个柱子的中心坐标固定在 0 和 1
        x1, x2 = 0, 1

        ax.plot([x1, x2], [line_y, line_y], lw=1.5, color='black')
        ax.plot([x1, x1], [line_y - 0.05, line_y], lw=1.5, color='black')
        ax.plot([x2, x2], [line_y - 0.05, line_y], lw=1.5, color='black')

        # 将具体 p 值写在线的中间
        ax.text((x1 + x2) / 2, text_y, p_text, ha='center', va='bottom', fontweight='bold', color='black', fontsize=14)

    # 【核心修改 4】：把基因名放到了图表的正上方作为大标题
    ax.set_title(f'{target}', fontsize=20, fontweight='bold', pad=15)

    # 清空 X 轴的总标题，只保留底下的刻度名(A23T, A23)
    ax.set_xlabel('')
    ax.set_ylabel('Relative Expression (Fold Change)', fontsize=14, fontweight='bold')

    # 放大底部的 A23T 和 A23 字体
    plt.xticks(fontsize=16, fontweight='bold')

    # 隐藏冗余的图例，让画面极度干净
    if ax.get_legend():
        ax.get_legend().remove()

    plt.tight_layout()
    plot_filename = f'qPCR_Expression_{target}.png'
    plt.savefig(plot_filename, dpi=300)
    plt.close()

# ================= 7. 导出原始数据与统计结果 =================
plot_data.to_csv('qPCR_Plot_Raw_Data.csv', index=False, encoding='utf-8-sig')

summary_data = []
for target in targets:
    for sample in ['A23T', 'A23']:
        group = plot_data[(plot_data['Target'] == target) & (plot_data['Sample'] == sample)]
        if not group.empty:
            summary_data.append({
                'Target (基因)': target,
                'Sample (样本)': sample,
                'Mean_Fold_Change (平均表达量)': round(group['Fold_Change'].mean(), 4),
                'SD (标准差)': round(group['Fold_Change'].std(), 4),
                'N (有效孔数)': len(group),
                'P-Value (vs A23T)': round(p_values.get(target, np.nan), 4) if sample != 'A23T' else 'Control (对照基准)'
            })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('qPCR_Summary_Statistics.csv', index=False, encoding='utf-8-sig')

print("\n✅ 数据处理完毕，纯净黑蓝版（带P值、上标题）批量出图已完成！")