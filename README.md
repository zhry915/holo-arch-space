# Holo-Arch Recognition Engine

基于全纯逻辑同构 (Holomorphic Logic Isomorphism) 与柯西 FFT 提取引擎的中国古建筑风格识别系统。

## 功能特性

- **全息雷达**: 上传任意中国古建筑图片，实时识别21种建筑风格流派
- **解析应力计算**: 使用 HoloPGF 引擎计算高阶分析应力 (Ψ₄)
- **水墨界画**: 利用 FFT 带通滤波生成传统界画/白描风格图像
- **模型解析**: 可视化训练过程中的损失地貌与优化走廊

## 支持的建筑风格

本系统支持识别以下 21 种中国古建筑风格：

1. 北京官式琉璃瓦
2. 北京四合院
3. 内蒙古传统毡房
4. 福建客家土楼
5. 江西赣南围屋
6. 山东胶东海草房
7. 河南地坑院
8. 湖南湘西吊脚楼
9. 广东岭南骑楼
10. 广西三江风雨桥
11. 云南傣族竹楼
12. 陕西陕北窑洞
13. 安徽徽派民居
14. 四川川西碉堡
15. 西藏藏式碉房
16. 苏式园林建筑
17. 新疆阿以旺式民居
18. 山西晋商大院
19. 湖北武当山古建筑群
20. 东北民居
21. 台湾闽南古厝

## 技术架构

- **后端框架**: FastAPI + Gradio
- **深度学习**: PyTorch + TorchVision
- **核心模型**: Holo-MobileNetV3 (MobileNetV3-Large with Holomorphic modules)
- **分析引擎**: HoloPGF Engine (基于柯西积分公式的高阶谱分析)

## 部署说明

### 方式一：直接部署到 Hugging Face Spaces

1. 将整个项目文件夹推送到 GitHub 仓库
2. 在 Hugging Face Spaces 创建新 Space，选择 **Gradio** SDK
3. 在 Space 设置中关联你的 GitHub 仓库
4. 系统将自动安装依赖并启动服务

### 方式二：本地运行

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python app.py

# 访问 http://127.0.0.1:7860
```

## 文件结构

```
HoloArch_HF_Space/
├── app.py                      # 主应用程序入口
├── index.html                  # 高端前端界面
├── requirements.txt            # Python 依赖
├── README.md                   # 项目说明文档
├── src/
│   ├── holo_mobilenet.py       # 核心模型定义
│   ├── holo_pgf_engine.py      # HoloPGF 分析引擎
│   ├── aesthetic_fft_sketch.py  # FFT 水墨界画生成
│   └── spectral_landscape_adamw_log.csv  # 训练日志数据
├── models/                     # 模型权重目录
│   └── mobilenet_v3_arch_spectral_best.pth
└── data/                       # 数据集目录
    └── arch_dataset_v1_cleaned.h5
```

## 重要提示

- **模型权重**: 请将训练好的模型权重文件 `mobilenet_v3_arch_spectral_best.pth` 放入 `models/` 目录
- **数据集**: 如果有 H5 格式的数据集，请放入 `data/` 目录（可选，系统有默认风格列表作为后备）
- **资源限制**: Hugging Face Spaces 有内存和存储限制，大文件请考虑使用外部存储

## 许可证

本项目仅供学术研究使用。

## 致谢

- 基于 MobileNetV3-Large 预训练模型
- 前端设计灵感来源于现代 Web 交互体验
