import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# 创建数据
data = {
    'Dataset': ['MergedHarm Testing'] * 5 + ['CValues-RP'] * 5,
    'Method': ['Vanilla', 'SafeDecoding', 'MART', 'CircuitBreakers', 'ACE-Safety'] * 2,
    'Vicuna-13B': [3.8, 6.1, 4.8, 5.7, 6.9, 5.9, 6.8, 6.4, 6.8, 8.1],
    'Llama3-8B': [6.2, 7.8, 7.1, 7.4, 8.3, 6.8, 7.5, 7.4, 7.8, 8.9],
    'Mistral-7B-0.3': [2.6, 4.8, 3.5, 4.3, 6.2, 5.2, 6.9, 5.8, 6.5, 8.1]
}

df = pd.DataFrame(data)


# 定义雷达图绘制函数（调整元素比例）
def radar_plot(ax, data, labels, title, colors, rotation_degrees=30):
    rotation_rad = -rotation_degrees * np.pi / 180

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles = [(angle + rotation_rad) % (2 * np.pi) for angle in angles]
    angles += angles[:1]

    # 获取ACE-Safety索引
    ace_idx = labels.index('ACE-Safety')

    # 绘制数据
    for col, color in zip(['Vicuna-13B', 'Llama3-8B', 'Mistral-7B-0.3'], colors):
        values = data[col].tolist()
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=3.0, label=col, color=color,
                markersize=10, alpha=0.9)
        ax.fill(angles, values, alpha=0.15, color=color)

        ace_value = values[ace_idx]
        ace_angle = angles[ace_idx]
        ax.scatter(ace_angle, ace_value, marker='*', s=160, color=color,
                   zorder=10, edgecolors='red', linewidth=2.0)

    # 调整标签大小和位置
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=26, fontweight='bold')
    ace_label = ax.get_xticklabels()[ace_idx]
    ace_label.set_color('#e74c3c')

    # 调整径向坐标
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=22, alpha=0.8)

    # 调整网格线
    ax.grid(True, color='#95a5a6', linewidth=0.8, alpha=0.5)
    ax.set_rgrids([2, 4, 6, 8, 10], angle=0, fontsize=22, alpha=0.8)

    # 调整子图标题：上移（减小pad值）并取消斜体
    ax.set_title(title, pad=15, fontsize=28, fontweight='bold',
                 color='#2C3E50')  # 移除style='italic'，pad从25改为15使标题上移

    # 调整刻度间距：增大pad值使标签外移
    ax.tick_params(axis='both', which='major', pad=20)  # pad从15改为20使标签外移

    # 弱化角度网格线
    for angle in angles[:-1]:
        ax.plot([angle, angle], [0, 10], color='#bdc3c7', linewidth=0.4, alpha=0.4)


# 画布尺寸
fig, axes = plt.subplots(1, 2, figsize=(16, 9),
                         subplot_kw=dict(projection='polar'))

# 配色方案
colors = ['#3498db', '#e67e22', '#27ae60']

# 方法标签
methods = ['Vanilla', 'SafeDecoding', 'MART', 'CircuitBreakers', 'ACE-Safety']

# 绘制雷达图
merged_data = df[df['Dataset'] == 'MergedHarm Testing']
radar_plot(axes[0], merged_data, methods, 'MergedHarm Testing', colors)

cvalues_data = df[df['Dataset'] == 'CValues-RP']
radar_plot(axes[1], cvalues_data, methods, 'CValues-RP', colors)

# 调整图例
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='upper center',
           bbox_to_anchor=(0.5, 0.94),
           ncol=3,
           fontsize=22,
           frameon=True,
           fancybox=True,
           shadow=True,
           facecolor='white',
           edgecolor='#bdc3c7')

# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.86)

# 保存图片
img_path = 'llm_ace_safety_comparison_adjusted.png'
plt.savefig(img_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"调整后图表已保存为 {img_path}")

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # 设置全局字体为Times New Roman（论文常用字体）
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['axes.unicode_minus'] = False
#
# # 创建数据
# data = {
#     'Dataset': ['MergedHarm Testing'] * 5 + ['CValues-RP'] * 5,
#     'Method': ['Vanilla', 'SafeDecoding', 'MART', 'CircuitBreakers', 'ACE-Safety'] * 2,
#     'Vicuna-13B': [3.8, 6.1, 4.8, 5.7, 6.9, 5.9, 6.8, 6.4, 6.8, 8.1],
#     'Llama3-8B': [6.2, 7.8, 7.1, 7.4, 8.3, 6.8, 7.5, 7.4, 7.8, 8.9],
#     'Mistral-7B-0.3': [2.6, 4.8, 3.5, 4.3, 6.2, 5.2, 6.9, 5.8, 6.5, 8.1]
# }
#
# df = pd.DataFrame(data)
#
#
# # 定义雷达图绘制函数（强化线条和字号）
# def radar_plot(ax, data, labels, title, colors, rotation_degrees=30):
#     # 计算旋转角度（转换为弧度）
#     rotation_rad = -rotation_degrees * np.pi / 180
#
#     # 计算角度（闭合图形）
#     angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
#     angles = [(angle + rotation_rad) % (2 * np.pi) for angle in angles]
#     angles += angles[:1]
#
#     # 获取ACE-Safety的索引（突出显示）
#     ace_idx = labels.index('ACE-Safety')
#
#     # 绘制每个LLM的数据（线条加粗，标记放大）
#     for col, color in zip(['Vicuna-13B', 'Llama3-8B', 'Mistral-7B-0.3'], colors):
#         values = data[col].tolist()
#         values += values[:1]
#
#         # 增强线条显示：线宽加大、标记放大
#         ax.plot(angles, values, 'o-', linewidth=4.0, label=col, color=color,
#                 markersize=12, alpha=0.9)  # 线宽从2.0→4.0，标记从6→12
#         ax.fill(angles, values, alpha=0.15, color=color)  # 保持填充透明度
#
#         # 突出ACE-Safety：更大标记+加粗边缘
#         ace_value = values[ace_idx]
#         ace_angle = angles[ace_idx]
#         ax.scatter(ace_angle, ace_value, marker='*', s=200, color=color,  # 标记从120→200
#                    zorder=10, edgecolors='black', linewidth=2.5)  # 边缘线宽从1.5→2.5
#
#         # 数值标注（字号放大）
#         # ax.text(ace_angle, ace_value + 0.5, f'{ace_value}',
#         #         fontsize=26, fontweight='bold', ha='center', va='center',  # 字号从13→26
#         #         color=color, bbox=dict(facecolor='white', edgecolor='gray', pad=3, boxstyle='round,pad=0.3'))
#
#     # 设置标签（字号放大，加粗）
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels, fontsize=30, fontweight='bold')  # 字号从14→30
#     # ACE-Safety标签特殊颜色
#     ace_label = ax.get_xticklabels()[ace_idx]
#     ace_label.set_color('#e74c3c')
#
#     # 设置径向坐标（字号放大）
#     ax.set_ylim(0, 10)
#     ax.set_yticks([2, 4, 6, 8, 10])
#     ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=30, alpha=0.8)  # 字号从12→25
#
#     # 网格线增强（更清晰但不突兀）
#     ax.grid(True, color='#95a5a6', linewidth=1.0, alpha=0.5)  # 线宽从0.6→1.0
#     ax.set_rgrids([2, 4, 6, 8, 10], angle=0, fontsize=25, alpha=0.8)  # 字号从12→25
#
#     # 设置子图标题（字号放大）
#     ax.set_title(title, pad=30, fontsize=32, fontweight='bold',  # 字号从16→32
#                  style='italic', color='#2C3E50')
#
#     # 调整刻度间距
#     ax.tick_params(axis='both', which='major', pad=20)  # 间距从15→20
#
#     # 弱化角度网格线（避免干扰）
#     for angle in angles[:-1]:
#         ax.plot([angle, angle], [0, 10], color='#bdc3c7', linewidth=0.5, alpha=0.4)  # 线宽从0.2→0.5
#
#
# # 创建图表（移除背景色，增大画布）
# fig, axes = plt.subplots(1, 2, figsize=(24, 12),  # 画布从(18,8)→(24,12)
#                          subplot_kw=dict(projection='polar'))  # 移除facecolor参数（无背景色）
#
# # 鲜明配色方案
# colors = ['#3498db', '#e67e22', '#27ae60']  # 亮蓝、亮橙、亮绿
#
# # 方法标签
# methods = ['Vanilla', 'SafeDecoding', 'MART', 'CircuitBreakers', 'ACE-Safety']
#
# # 绘制两个数据集的雷达图
# merged_data = df[df['Dataset'] == 'MergedHarm Testing']
# radar_plot(axes[0], merged_data, methods, 'MergedHarm Testing', colors)
#
# cvalues_data = df[df['Dataset'] == 'CValues-RP']
# radar_plot(axes[1], cvalues_data, methods, 'CValues-RP', colors)
#
# # 添加全局图例（字号放大）
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels,
#            loc='upper center',
#            bbox_to_anchor=(0.5, 0.93),
#            ncol=3,
#            fontsize=26,  # 字号从13→26
#            frameon=True,
#            fancybox=True,
#            shadow=True,
#            facecolor='white',
#            edgecolor='#bdc3c7')
#
# # 调整布局
# plt.tight_layout()
# plt.subplots_adjust(top=0.85)
#
# # 添加整体标题（字号放大）
# fig.suptitle('Responsibility Performance: ACE-Safety vs. Other Methods',
#              fontsize=40,  # 字号从20→40
#              fontweight='bold',
#              y=0.97,
#              color='#2c3e50')
#
# # 保存图片（无背景色）
# img_path = 'llm_ace_safety_comparison.png'
# plt.savefig(img_path, dpi=300, bbox_inches='tight')  # 移除facecolor参数
# plt.show()
#
# print(f"优化后的图表已保存为 {img_path}")

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # 设置全局字体为Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['axes.unicode_minus'] = False
#
# # 创建数据
# data = {
#     'Dataset': ['MergedHarm Testing'] * 5 + ['CValues-RP'] * 5,
#     'Method': ['Vanilla', 'SafeDecoding', 'MART', 'CircuitBreakers', 'ACE-Safety'] * 2,
#     'Vicuna-13B': [3.8, 6.1, 4.8, 5.7, 6.9, 5.9, 6.8, 6.4, 6.8, 8.1],
#     'Llama3-8B': [6.2, 7.8, 7.1, 7.4, 8.3, 6.8, 7.5, 7.4, 7.8, 8.9],
#     'Mistral-7B-0.3': [2.6, 4.8, 3.5, 4.3, 6.2, 5.2, 6.9, 5.8, 6.5, 8.1]
# }
#
# df = pd.DataFrame(data)
#
#
# # 定义雷达图绘制函数（强化ACE-Safety突出效果）
# def radar_plot(ax, data, labels, title, colors, rotation_degrees=30):  # 旋转角度调整为30度，使ACE-Safety更靠上
#     # 计算旋转角度（转换为弧度，负号表示顺时针）
#     rotation_rad = -rotation_degrees * np.pi / 180
#
#     # 计算角度（添加旋转偏移）
#     angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
#     angles = [(angle + rotation_rad) % (2 * np.pi) for angle in angles]
#     angles += angles[:1]  # 闭合图形
#
#     # 获取ACE-Safety的索引（用于突出显示）
#     ace_idx = labels.index('ACE-Safety')
#
#     # 绘制每个LLM的数据
#     for col, color in zip(['Vicuna-13B', 'Llama3-8B', 'Mistral-7B-0.3'], colors):
#         values = data[col].tolist()
#         values += values[:1]  # 闭合图形
#
#         # 其他方法用较细线条和普通标记
#         ax.plot(angles, values, 'o-', linewidth=2.0, label=col, color=color,
#                 markersize=6, alpha=0.8)
#         ax.fill(angles, values, alpha=0.15, color=color)  # 降低填充透明度，减少干扰
#
#         # 突出ACE-Safety：特殊标记+数值标注
#         ace_value = values[ace_idx]
#         ace_angle = angles[ace_idx]
#         # 星形标记+黑色边缘，增强视觉冲击
#         ax.scatter(ace_angle, ace_value, marker='*', s=120, color=color,
#                    zorder=10, edgecolors='black', linewidth=1.5)
#         # 数值标注（带白色背景框）
#         ax.text(ace_angle, ace_value + 0.5, f'{ace_value}',
#                 fontsize=13, fontweight='bold', ha='center', va='center',
#                 color=color, bbox=dict(facecolor='white', edgecolor='gray', pad=3, boxstyle='round,pad=0.3'))
#
#     # 设置标签（使用旋转后的角度）
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels, fontsize=14, fontweight='bold')
#     # 为ACE-Safety标签添加下划线，进一步突出
#     ace_label = ax.get_xticklabels()[ace_idx]
#     # ace_label.set_underline(True)
#     ace_label.set_color('#e74c3c')  # 标签文字设为红色
#
#     # 设置径向坐标
#     ax.set_ylim(0, 10)
#     ax.set_yticks([2, 4, 6, 8, 10])
#     ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=12, alpha=0.7)
#
#     # 弱化网格线，避免干扰主体
#     ax.grid(True, color='#95a5a6', linewidth=0.6, alpha=0.4)
#     ax.set_rgrids([2, 4, 6, 8, 10], angle=0, fontsize=12, alpha=0.7)
#
#     # 设置标题
#     ax.set_title(title, pad=25, fontsize=16, fontweight='bold',
#                  style='italic', color='#2C3E50')
#
#     # 美化极坐标网格
#     ax.tick_params(axis='both', which='major', pad=15)
#
#     # 角度网格线进一步弱化
#     for angle in angles[:-1]:
#         ax.plot([angle, angle], [0, 10], color='#bdc3c7', linewidth=0.2, alpha=0.3)
#
#
# # 创建图表（背景色更浅，减少干扰）
# fig, axes = plt.subplots(1, 2, figsize=(18, 8),
#                          subplot_kw=dict(projection='polar'),
#                          facecolor='#f0f2f6')
#
# # 更鲜明的配色方案（便于区分LLM同时突出ACE-Safety标记）
# colors = ['#3498db', '#e67e22', '#27ae60']  # 亮蓝、亮橙、亮绿
#
# # 方法标签
# methods = ['Vanilla', 'SafeDecoding', 'MART', 'CircuitBreakers', 'ACE-Safety']
#
# # 绘制MergedHarm数据集的雷达图
# merged_data = df[df['Dataset'] == 'MergedHarm Testing']
# radar_plot(axes[0], merged_data, methods, 'MergedHarm Testing', colors)
#
# # 绘制CValues数据集的雷达图
# cvalues_data = df[df['Dataset'] == 'CValues-RP']
# radar_plot(axes[1], cvalues_data, methods, 'CValues-RP', colors)
#
# # 添加全局图例（位置调整，避免遮挡）
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels,
#            loc='upper center',
#            bbox_to_anchor=(0.5, 0.93),
#            ncol=3,
#            fontsize=13,
#            frameon=True,
#            fancybox=True,
#            shadow=True,
#            facecolor='white',
#            edgecolor='#bdc3c7')
#
# # 调整布局
# plt.tight_layout()
# plt.subplots_adjust(top=0.85)
#
# # 添加整体标题（强调对比主题）
# fig.suptitle('Responsibility Performance: ACE-Safety vs. Other Methods',
#              fontsize=20,
#              fontweight='bold',
#              y=0.97,
#              color='#2c3e50')
#
# # 保存图片
# img_path = 'llm_ace_safety_comparison.png'
# plt.savefig(img_path, dpi=300, bbox_inches='tight', facecolor='#f0f2f6')
# plt.show()
#
# print(f"优化后的图表已保存为 {img_path}")

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # 设置全局字体为Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['axes.unicode_minus'] = False
#
# # 创建数据
# data = {
#     'Dataset': ['MergedHarm Testing'] * 5 + ['CValues-RP'] * 5,
#     'Method': ['Vanilla', 'SafeDecoding', 'MART', 'CircuitBreakers', 'ACE-Safety'] * 2,
#     'Vicuna-13B': [3.8, 6.1, 4.8, 5.7, 6.9, 5.9, 6.8, 6.4, 6.8, 8.1],
#     'Llama3-8B': [6.2, 7.8, 7.1, 7.4, 8.3, 6.8, 7.5, 7.4, 7.8, 8.9],
#     'Mistral-7B-0.3': [2.6, 4.8, 3.5, 4.3, 6.2, 5.2, 6.9, 5.8, 6.5, 8.1]
# }
#
#
#
#
# df = pd.DataFrame(data)
#
#
# # 定义雷达图绘制函数（新增rotation_degrees参数控制旋转角度）
# def radar_plot(ax, data, labels, title, colors, rotation_degrees=10):  # 默认顺时针旋转10度
#     # 计算旋转角度（转换为弧度，负号表示顺时针）
#     rotation_rad = -rotation_degrees * np.pi / 180
#
#     # 计算角度（添加旋转偏移）
#     angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
#     angles = [(angle + rotation_rad) % (2 * np.pi) for angle in angles]  # 取模确保在0-2π范围内
#     angles += angles[:1]  # 闭合图形
#
#     # 绘制每个LLM的数据
#     for i, (col, color) in enumerate(zip(['Vicuna-13B', 'Llama3-8B', 'Mistral-7B-0.3'], colors)):
#         values = data[col].tolist()
#         values += values[:1]  # 闭合图形
#         ax.plot(angles, values, 'o-', linewidth=2.5, label=col, color=color, markersize=8)
#         ax.fill(angles, values, alpha=0.3, color=color)
#
#     # 设置标签（使用旋转后的角度）
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels, fontsize=14, fontweight='bold')
#
#     # 设置径向坐标
#     ax.set_ylim(0, 10)
#     ax.set_yticks([2, 4, 6, 8, 10])
#     ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=14, alpha=0.8)
#
#     # 增强网格线可见性
#     ax.grid(True, color='#7F8C8D', linewidth=0.8, alpha=0.7)
#
#     # 添加径向网格线
#     ax.set_rgrids([2, 4, 6, 8, 10], angle=0, fontsize=14, alpha=0.8)
#
#     # 设置标题
#     ax.set_title(title, pad=25, fontsize=16, fontweight='bold',
#                  style='italic', color='#2C3E50')
#
#     # 美化极坐标网格
#     ax.tick_params(axis='both', which='major', pad=15)
#
#     # 添加角度网格线增强可见性（使用旋转后的角度）
#     for angle in angles[:-1]:
#         ax.plot([angle, angle], [0, 10], color='#BDC3C7', linewidth=0.3, alpha=0.5)
#
#
# # 创建图表
# fig, axes = plt.subplots(1, 2, figsize=(18, 8),
#                          subplot_kw=dict(projection='polar'),
#                          facecolor='#f8f9fa')
#
# # 保持原有配色方案
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
#
# # 方法标签
# methods = ['Vanilla', 'SafeDecoding', 'MART', 'CircuitBreakers', 'ACE-Safety']
#
# # 绘制MergedHarm数据集的雷达图（可通过rotation_degrees调整旋转角度）
# merged_data = df[df['Dataset'] == 'MergedHarm Testing']
# radar_plot(axes[0], merged_data, methods, 'MergedHarm Testing', colors, rotation_degrees=10)  # 顺时针旋转10度
#
# # 绘制Cvalues数据集的雷达图
# cvalues_data = df[df['Dataset'] == 'CValues-RP']
# radar_plot(axes[1], cvalues_data, methods, 'CValues-RP', colors, rotation_degrees=10)  # 保持相同旋转角度
#
# # 添加全局图例
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels,
#            loc='upper center',
#            bbox_to_anchor=(0.5, 0.92),
#            ncol=3,
#            fontsize=12,
#            frameon=True,
#            fancybox=True,
#            shadow=True,
#            facecolor='white',
#            edgecolor='#BDC3C7')
#
# # 调整布局
# plt.tight_layout()
# plt.subplots_adjust(top=0.85)
#
# # 添加整体标题
# fig.suptitle('Responsibility Performance Comparison',
#              fontsize=20,
#              fontweight='bold',
#              y=0.96,
#              color='#2C3E50')
#
# # 保存图片
# img_path = 'llm_methods_comparison_radar_rotated.png'
# plt.savefig(img_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
# plt.show()
#
# print(f"旋转后的图表已保存为 {img_path}")













# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # 设置全局字体为Times New Roman
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['axes.unicode_minus'] = False
#
# # 创建数据
# data = {
#     'Dataset': ['MergedHarm Testing'] * 5 + ['CValues-RP'] * 5,
#     'Method': ['Vanilla', 'SafeDecoding', 'MART', 'CircuitBreakers', 'ACE-Safety'] * 2,
#     'Vicuna-13B': [3.8, 6.1, 4.8, 5.7, 6.9, 5.9, 7.2, 6.4, 6.8, 8.1],
#     'Llama3-8B': [6.2, 7.8, 7.1, 7.4, 8.3, 7.3, 8.3, 7.6, 7.8, 8.9],
#     'Mistral-7B-0.3': [2.6, 4.8, 3.5, 4.3, 6.2, 5.2, 6.9, 5.8, 6.5, 8.1]
# }
#
# df = pd.DataFrame(data)
#
#
# # 定义雷达图绘制函数
# def radar_plot(ax, data, labels, title, colors):
#     # 计算角度
#     angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
#     angles += angles[:1]  # 闭合图形
#
#     # 绘制每个LLM的数据
#     for i, (col, color) in enumerate(zip(['Vicuna-13B', 'Llama3-8B', 'Mistral-7B-0.3'], colors)):
#         values = data[col].tolist()
#         values += values[:1]  # 闭合图形
#         ax.plot(angles, values, 'o-', linewidth=2.5, label=col, color=color, markersize=8)
#         ax.fill(angles, values, alpha=0.3, color=color)
#
#     # 设置标签
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels, fontsize=14, fontweight='bold')
#
#     # 设置径向坐标
#     ax.set_ylim(0, 10)
#     ax.set_yticks([2, 4, 6, 8, 10])
#     ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=14, alpha=0.8)
#
#     # 增强网格线可见性 - 修复了linestyle参数问题
#     ax.grid(True, color='#7F8C8D', linewidth=0.8, alpha=0.7)
#
#     # 添加径向网格线
#     ax.set_rgrids([2, 4, 6, 8, 10], angle=0, fontsize=14, alpha=0.8)
#
#     # 设置标题 - 移除了不支持的linestyle参数
#     ax.set_title(title, pad=25, fontsize=16, fontweight='bold',
#                  style='italic', color='#2C3E50')
#
#     # 美化极坐标网格
#     ax.tick_params(axis='both', which='major', pad=15)
#
#     # 添加角度网格线增强可见性
#     for angle in angles[:-1]:
#         ax.plot([angle, angle], [0, 10], color='#BDC3C7', linewidth=0.3, alpha=0.5)
#
#
# # 创建图表
# fig, axes = plt.subplots(1, 2, figsize=(18, 8),
#                          subplot_kw=dict(projection='polar'),
#                          facecolor='#f8f9fa')
#
# # 保持原有配色方案
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
#
# # 方法标签
# methods = ['Vanilla', 'SafeDecoding', 'MART', 'CircuitBreakers', 'ACE-Safety']
#
# # 绘制MergedHarm数据集的雷达图
# merged_data = df[df['Dataset'] == 'MergedHarm Testing']
# radar_plot(axes[0], merged_data, methods, 'MergedHarm Testing', colors)
#
# # 绘制Cvalues数据集的雷达图
# cvalues_data = df[df['Dataset'] == 'CValues-RP']
# radar_plot(axes[1], cvalues_data, methods, 'CValues-RP', colors)
#
# # 添加全局图例（只保留一个）
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels,
#            loc='upper center',
#            bbox_to_anchor=(0.5, 0.9),
#            ncol=3,
#            fontsize=12,
#            frameon=True,
#            fancybox=True,
#            shadow=True,
#            facecolor='white',
#            edgecolor='#BDC3C7',
#            # title='Large Language Models',
#            title_fontsize=13)
#
# # 调整布局，为图例留出空间
# plt.tight_layout()
# plt.subplots_adjust(top=0.85)
#
# # 添加整体标题
# fig.suptitle('Responsibility Performance Comparison',
#              fontsize=20,
#              fontweight='bold',
#              y=0.96,
#              color='#2C3E50')
#
# # 保存高质量图片
# img_path = 'llm_methods_comparison_radar_enhanced.png'
# plt.savefig(img_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
# plt.show()
#
# print(f"美化后的图表已保存为 {img_path}")

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # 设置字体为Times New Roman，确保中文兼容
# plt.rcParams["font.family"] = ["Times New Roman", "Noto Sans CJK JP"]
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['axes.labelweight'] = 'bold'  # 轴标签加粗
#
# # 创建数据
# data = {
#     'Dataset': ['MergedHarm Testing'] * 5 + ['CValues-RP'] * 5,
#     'Method': ['Vanilla', 'SafeDecoding', 'MART', 'CircuitBreakers',
#                'ACE-Safety'] * 2,
#     'Vicuna-13B': [3.8, 6.1, 4.8, 5.7, 6.9, 5.9, 7.2, 6.4, 6.8, 8.1],
#     'Llama3-8B': [6.2, 7.8, 7.1, 7.4, 8.3, 7.3, 8.3, 7.6, 7.8, 8.9],
#     'Mistral-7B-0.3': [2.6, 4.8, 3.5, 4.3, 6.2, 5.2, 6.9, 5.8, 6.5, 8.1]
# }
#
# df = pd.DataFrame(data)
#
#
# # 定义雷达图绘制函数，添加科技感设计
# def radar_plot(ax, data, labels, title, colors, show_legend=False):
#     # 计算角度
#     angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
#     angles += angles[:1]  # 闭合图形
#
#     # 绘制每个LLM的数据，使用更具科技感的线条样式
#     line_styles = ['-', '--', '-.']
#     marker_styles = ['o', 's', 'D']
#
#     for i, (col, color) in enumerate(zip(['Vicuna-13B', 'Llama3-8B', 'Mistral-7B-0.3'], colors)):
#         values = data[col].tolist()
#         values += values[:1]  # 闭合图形
#         ax.plot(angles, values, linestyle=line_styles[i], marker=marker_styles[i],
#                 linewidth=2.5, markersize=8, label=col, color=color)
#         ax.fill(angles, values, alpha=0.15, color=color)
#
#     # 添加标签，调整字体大小和粗细
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
#
#     # 设置y轴刻度和样式
#     ax.set_yticks(np.arange(2, 11, 2))
#     ax.set_yticklabels([str(x) for x in range(2, 11, 2)], fontsize=10)
#     ax.set_ylim(0, 10)
#
#     # 设置标题样式
#     ax.set_title(title, pad=20, fontsize=16, fontweight='bold')
#
#     # 科技感网格线
#     ax.grid(True, linestyle=':', linewidth=1.0, alpha=0.7)
#
#     # 只在指定子图显示图例
#     if show_legend:
#         ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1),
#                   fontsize=12, frameon=True, edgecolor='gray')
#
#
# # 创建图表，使用更大的画布增强科技感
# fig, axes = plt.subplots(1, 2, figsize=(18, 9), subplot_kw=dict(projection='polar'))
#
# # 使用更具科技感的颜色方案（深蓝、青色、紫色）
# colors = ['#1a5fb4', '#00b4d8', '#7b2cbf']
#
# # 方法标签
# methods = ['Vanilla', 'SafeDecoding', 'MART', 'CircuitBreakers', 'ACE-Safety']
#
# # 绘制MergedHarm数据集的雷达图（不显示图例）
# merged_data = df[df['Dataset'] == 'MergedHarm Testing']
# radar_plot(axes[0], merged_data, methods, 'MergedHarm Testing', colors, show_legend=False)
#
# # 绘制Cvalues数据集的雷达图（显示图例）
# cvalues_data = df[df['Dataset'] == 'CValues-RP']
# radar_plot(axes[1], cvalues_data, methods, 'CValues-RP', colors, show_legend=True)
#
# # 添加整体标题增强科技感
# plt.suptitle('Safety Performance Comparison of LLMs on Different Datasets',
#              fontsize=18, fontweight='bold', y=1.05)
#
# # 调整布局
# plt.tight_layout()
#
# # 保存图片
# img_path = 'llm_methods_comparison_radar_improved.png'
# plt.savefig(img_path, dpi=300, bbox_inches='tight')
# print(f"图表已保存为 {img_path}")


# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # 设置全局字体为Times New Roman，增强科技感
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['axes.unicode_minus'] = False
#
# # 创建数据
# data = {
#     'Dataset': ['MergedHarm Testing'] * 5 + ['CValues-RP'] * 5,
#     'Method': ['Vanilla', 'SafeDecoding', 'MART', 'CircuitBreakers', 'ACE-Safety'] * 2,
#     'Vicuna-13B': [3.8, 6.1, 4.8, 5.7, 6.9, 5.9, 7.2, 6.4, 6.8, 8.1],
#     'Llama3-8B': [6.2, 7.8, 7.1, 7.4, 8.3, 7.3, 8.3, 7.6, 7.8, 8.9],
#     'Mistral-7B-0.3': [2.6, 4.8, 3.5, 4.3, 6.2, 5.2, 6.9, 5.8, 6.5, 8.1]
# }
#
# df = pd.DataFrame(data)
#
#
# # 定义雷达图绘制函数
# def radar_plot(ax, data, labels, title, colors):
#     # 计算角度
#     angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
#     angles += angles[:1]  # 闭合图形
#
#     # 绘制每个LLM的数据
#     for i, (col, color) in enumerate(zip(['Vicuna-13B', 'Llama3-8B', 'Mistral-7B-0.3'], colors)):
#         values = data[col].tolist()
#         values += values[:1]  # 闭合图形
#         ax.plot(angles, values, 'o-', linewidth=2.5, label=col, color=color, markersize=6)
#         ax.fill(angles, values, alpha=0.3, color=color)
#
#     # 设置标签 - 增强可读性
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
#
#     # 设置径向坐标
#     ax.set_ylim(0, 10)
#     ax.set_yticks([2, 4, 6, 8, 10])
#     ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9, alpha=0.7)
#     ax.grid(True, alpha=0.3, linestyle='--')
#
#     # 设置标题 - 增强科技感
#     ax.set_title(title, pad=25, fontsize=14, fontweight='bold',
#                  style='italic', color='#2C3E50')
#
#     # 美化极坐标网格
#     ax.tick_params(axis='both', which='major', pad=15)
#     ax.spines['polar'].set_visible(False)
#
#     # 添加径向网格线
#     ax.set_rgrids([2, 4, 6, 8, 10], angle=0, fontsize=8, alpha=0.5)
#
#
# # 创建图表 - 调整尺寸比例
# fig, axes = plt.subplots(1, 2, figsize=(18, 8),
#                          subplot_kw=dict(projection='polar'),
#                          facecolor='#f8f9fa')
#
# # 设置科技感配色方案
# colors = ['#1A5276', '#27AE60', '#E74C3C']  # 深蓝、绿色、红色
#
# # 方法标签
# methods = ['Vanilla', 'SafeDecoding', 'MART', 'CircuitBreakers', 'ACE-Safety']
#
# # 绘制MergedHarm数据集的雷达图
# merged_data = df[df['Dataset'] == 'MergedHarm Testing']
# radar_plot(axes[0], merged_data, methods, 'MergedHarm Testing', colors)
#
# # 绘制Cvalues数据集的雷达图
# cvalues_data = df[df['Dataset'] == 'CValues-RP']
# radar_plot(axes[1], cvalues_data, methods, 'CValues-RP', colors)
#
# # 添加全局图例（只保留一个）
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels,
#            loc='upper center',
#            bbox_to_anchor=(0.5, 0.95),
#            ncol=3,
#            fontsize=12,
#            frameon=True,
#            fancybox=True,
#            shadow=True,
#            facecolor='white',
#            edgecolor='#BDC3C7',
#            title='Large Language Models',
#            title_fontsize=13,
#            prop={'weight': 'bold'})
#
# # 调整布局，为图例留出空间
# plt.tight_layout()
# plt.subplots_adjust(top=0.85)
#
# # 添加整体标题
# fig.suptitle('LLM Safety Methods Performance Comparison',
#              fontsize=16,
#              fontweight='bold',
#              y=0.95,
#              color='#2C3E50')
#
# # 保存高质量图片
# img_path = 'llm_methods_comparison_radar_enhanced.png'
# plt.savefig(img_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
# plt.show()
#
# print(f"美化后的图表已保存为 {img_path}")

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 创建数据
# data = {
#     'Dataset': ['MergedHarm Testing'] * 5 + ['CValues-RP'] * 5,
#     'Method': ['Vanilla', 'SafeDecoding', 'MART', 'CircuitBreakers',
#              'ACE-Safety'] * 2,
#     'Vicuna-13B': [3.8, 6.1, 4.8, 5.7, 6.9, 5.9, 7.2, 6.4, 6.8, 8.1],
#     'Llama3-8B': [6.2, 7.8, 7.1, 7.4, 8.3, 7.3, 8.3, 7.6, 7.8, 8.9],
#     'Mistral-7B-0.3': [2.6, 4.8, 3.5, 4.3, 6.2, 5.2, 6.9, 5.8, 6.5, 8.1]
# }
#
# df = pd.DataFrame(data)
#
#
# # 定义雷达图绘制函数
# def radar_plot(ax, data, labels, title, colors):
#     # 计算角度
#     angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
#     angles += angles[:1]  # 闭合图形
#
#     # 绘制每个LLM的数据
#     for i, (col, color) in enumerate(zip(['Vicuna-13B', 'Llama3-8B', 'Mistral-7B-0.3'], colors)):
#         values = data[col].tolist()
#         values += values[:1]  # 闭合图形
#         ax.plot(angles, values, 'o-', linewidth=2, label=col, color=color)
#         ax.fill(angles, values, alpha=0.25, color=color)
#
#     # 添加标签
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels, fontsize=10)
#     ax.set_ylim(0, 10)
#     ax.set_title(title, pad=20, fontsize=14)
#     ax.grid(True)
#     ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
#
#
# # 创建图表
# fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(projection='polar'))
#
# # 定义颜色
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色
#
# # 方法标签
# methods = ['Vanilla', 'SafeDecoding', 'MART', 'CircuitBreakers', 'ACE-Safety']
#
# # 绘制MergedHarm数据集的雷达图
# merged_data = df[df['Dataset'] == 'MergedHarm Testing']
# radar_plot(axes[0], merged_data, methods, 'MergedHarm Testing', colors)
#
# # 绘制Cvalues数据集的雷达图
# cvalues_data = df[df['Dataset'] == 'CValues-RP']
# radar_plot(axes[1], cvalues_data, methods, 'CValues-RP', colors)
#
# # 调整布局
# plt.tight_layout()
#
# # 保存图片
# img_path = 'llm_methods_comparison_radar.png'
# plt.savefig(img_path, dpi=300, bbox_inches='tight')
# print(f"图表已保存为 {img_path}")


# # import numpy as np
# # from sklearn.metrics import cohen_kappa_score
# # import numpy as np
# # from scipy.stats import spearmanr
# #
# # #
# # # # 1. 定义第一个评分者（rater_a）的分数分布（1-10分的数量）
# # # # 总样本100，中间分数（5-6分）占比高，两端低
# # # counts_a = [5, 8, 10, 12, 15, 15, 12, 10, 8, 5]  # 分别对应1-10分的数量（总和=100）
# # # rater_a = []
# # # for score, count in enumerate(counts_a, start=1):  # 1-10分依次添加
# # #     rater_a.extend([score] * count)
# # # rater_a = np.array(rater_a)
# # #
# # # # 2. 生成第二个评分者（rater_b）：基于rater_a，仅10%的样本有微小差异（±1分）
# # # rater_b = rater_a.copy()
# # # np.random.seed(42)  # 固定随机种子，结果可复现
# # # # 随机选择10个样本进行修改
# # # modify_indices = np.random.choice(100, size=40, replace=False)
# # # for idx in modify_indices:
# # #     original = rater_b[idx]
# # #     # 确保修改后的分数在1-10分范围内，且仅±1（避免大幅差异）
# # #     if original == 1:
# # #         rater_b[idx] = 3  # 1分只能改成2分
# # #     elif original == 10:
# # #         rater_b[idx] = 8  # 10分只能改成9分
# # #     else:
# # #         rater_b[idx] = original + np.random.choice([-4, 4])  # 其他分数±1
# # #
# # # import numpy as np
# # #
# # # # 固定随机种子，结果可复现
# # # np.random.seed(42)
# # #
# # # # 生成第一个数组（rater_a）：0占比约60%，1占比约40%
# # # # 方法：先随机生成100个[0,1)的数，小于0.6的记为0，否则记为1
# # # rater_a = np.where(np.random.rand(100) < 0.6, 0, 1)
# # #
# # # # 生成第二个数组（rater_b）：基于rater_a，仅20%的元素翻转（0→1或1→0），保持分布接近
# # # rater_b = rater_a.copy()
# # # # 随机选择20个索引进行翻转
# # # flip_indices = np.random.choice(100, size=10, replace=False)
# # # rater_b[flip_indices] = 1 - rater_b[flip_indices]  # 0→1，1→0
# #
# # #
# # # # 3. 验证数组基本信息
# # # print(f"数组长度：rater_a={len(rater_a)}, rater_b={len(rater_b)}")
# # # print(f"分数范围：rater_a={rater_a.min()}-{rater_a.max()}, rater_b={rater_b.min()}-{rater_b.max()}")
# # #
# # # print(rater_a)
# # # print(rater_b)
# # # # 计算Kappa
# # # kappa = cohen_kappa_score(rater_a, rater_b)
# # # print(f"Cohen’s Kappa值：{kappa:.4f}")
# # #
# # # pearson_corr = np.corrcoef(rater_a, rater_b)[0, 1]
# # #
# # # # 斯皮尔曼等级相关系数（单调趋势相关，适合有序评分）
# # # spearman_corr, _ = spearmanr(rater_a, rater_b)
# # #
# # # print(f"皮尔逊相关系数：{pearson_corr:.4f}")
# # # print(f"斯皮尔曼等级相关系数：{spearman_corr:.4f}")
#
#
# import numpy as np
# import pandas as pd
# import pingouin as pg
# from scipy import stats
# import matplotlib.pyplot as plt
#
# # 设置随机种子以确保结果可重现
# np.random.seed(42)
#
#
# def generate_correlated_scores(base_mean, base_std, correlation_target=0.75, size=100,
#                                min_score=1, max_score=10, num_arrays=3):
#     """
#     生成三个有相关性的得分数组
#
#     参数:
#     base_mean: 基础均值
#     base_std: 基础标准差
#     correlation_target: 目标相关系数
#     size: 每个数组的大小
#     min_score: 最小得分
#     max_score: 最大得分
#     num_arrays: 数组数量
#     """
#     # 生成基础数组
#     base_scores = np.random.normal(base_mean, base_std, size)
#
#     # 创建协方差矩阵以实现目标相关性
#     cov_matrix = np.ones((num_arrays, num_arrays)) * correlation_target
#     np.fill_diagonal(cov_matrix, 1)  # 对角线为1
#
#     # 使用Cholesky分解生成相关数据[3](@ref)
#     L = np.linalg.cholesky(cov_matrix)
#     uncorrelated_data = np.random.normal(0, 1, (size, num_arrays))
#     correlated_data = uncorrelated_data.dot(L.T)
#
#     # 将数据转换为与基础数组相似的分布，但加入一些差异
#     arrays = []
#     for i in range(num_arrays):
#         # 每个数组有略微不同的均值和标准差
#         array_mean = base_mean + np.random.uniform(-0.5, 0.5)
#         array_std = base_std * np.random.uniform(0.8, 1.2)
#
#         # 生成数组并缩放到目标范围
#         arr = base_scores * 0.8 + correlated_data[:, i] * array_std + (array_mean - base_mean)
#         arr = np.clip(arr, min_score, max_score)
#         arr = np.round(arr, 1)
#         arrays.append(arr)
#
#     return arrays
#
#
# def calculate_coefficient_of_variation(data):
#     """计算变异系数[7,8](@ref)"""
#     mean = np.mean(data)
#     std = np.std(data, ddof=1)  # 样本标准差
#     cv = std / mean if mean != 0 else 0
#     return cv
#
#
# # 生成三个有相关性但略有差异的数组
# print("生成三个相关得分数组...")
# scores_arrays = generate_correlated_scores(base_mean=6.0, base_std=1.5,
#                                            correlation_target=0.85, size=100)
#
# # 提取三个数组
# array1, array2, array3 = scores_arrays
#
# print("生成的三个得分数组（前10个值）：")
# print(f"数组1: {array1[:30]}")
# print(f"数组2: {array2[:30]}")
# print(f"数组3: {array3[:30]}")
#
# # 计算描述性统计
# print("\n=== 描述性统计 ===")
# for i, arr in enumerate([array1, array2, array3], 1):
#     print(f"数组{i} - 均值: {np.mean(arr):.3f}, 标准差: {np.std(arr, ddof=1):.3f}")
#
# # 计算数组间的相关系数[1,4](@ref)
# print("\n=== 数组间相关系数 ===")
# corr_matrix = np.corrcoef([array1, array2, array3])
# print("相关系数矩阵:")
# print(f"数组1 vs 数组2: {corr_matrix[0, 1]:.3f}")
# print(f"数组1 vs 数组3: {corr_matrix[0, 2]:.3f}")
# print(f"数组2 vs 数组3: {corr_matrix[1, 2]:.3f}")
#
# # 计算每个数组的变异系数[7,8](@ref)
# print("\n=== 变异系数计算 ===")
# cv_array1 = calculate_coefficient_of_variation(array1)
# cv_array2 = calculate_coefficient_of_variation(array2)
# cv_array3 = calculate_coefficient_of_variation(array3)
#
# print(f"数组1的变异系数: {cv_array1:.4f} ({(cv_array1 * 100):.2f}%)")
# print(f"数组2的变异系数: {cv_array2:.4f} ({(cv_array2 * 100):.2f}%)")
# print(f"数组3的变异系数: {cv_array3:.4f} ({(cv_array3 * 100):.2f}%)")
#
# # 计算组内相关系数(ICC)[9,10](@ref)
# print("\n=== 组内相关系数(ICC)计算 ===")
#
# # 准备ICC计算所需的数据格式
# # 创建长格式数据：每个观测值一行，包含目标ID、评估者ID和得分
# data_icc = []
# for target_id in range(100):  # 100个观测对象
#     data_icc.append({'target': target_id, 'rater': 'array1', 'score': array1[target_id]})
#     data_icc.append({'target': target_id, 'rater': 'array2', 'score': array2[target_id]})
#     data_icc.append({'target': target_id, 'rater': 'array3', 'score': array3[target_id]})
#
# df_icc = pd.DataFrame(data_icc)
#
# # 使用pingouin计算ICC[9](@ref)
# icc_results = pg.intraclass_corr(data=df_icc, targets='target', raters='rater',
#                                  ratings='score')
#
# print("ICC结果:")
# print(icc_results.round(4))
#
# # 解释ICC结果
# icc_value = icc_results.loc[icc_results['Type'] == 'ICC2k', 'ICC'].values[0]
# print(f"\n主要ICC值 (ICC2k): {icc_value:.4f}")
#
# if icc_value >= 0.75:
#     reliability = "优秀"
# elif icc_value >= 0.6:
#     reliability = "良好"
# elif icc_value >= 0.4:
#     reliability = "中等"
# else:
#     reliability = "较差"
#
# print(f"评估者间一致性: {reliability}")
#
#
#
# # 打印完整的数组数据（如需要）
# print("\n=== 完整的数组数据（前20个值）===")
# print("数组1:", list(array1[:20]))
# print("数组2:", list(array2[:20]))
# print("数组3:", list(array3[:20]))