# Logistic Regression 演示

这是一个使用 React 和 TypeScript 实现的 Logistic Regression（逻辑回归）演示项目。该项目展示了如何使用逻辑回归算法预测学生的考试通过情况。

## 功能特点

- 使用真实数据进行模型训练
- 可视化训练过程中的损失函数变化
- 实时显示模型参数的更新
- 交互式预测界面
- 数据点可视化展示

## 项目结构

```
logistic-regression-demo/
├── public/
│   └── data.csv         # 训练数据
├── src/
│   ├── components/      # React组件
│   ├── App.tsx         # 主应用组件
│   ├── main.tsx        # 入口文件
│   └── *.css           # 样式文件
└── package.json        # 项目配置
```

## 安装和运行

1. 安装依赖：

```bash
npm install
```

2. 启动开发服务器：

```bash
npm run dev
```

3. 访问演示页面：

打开浏览器访问 http://localhost:5173

## 数据说明

训练数据包含以下字段：
- 每周学习时间（小时）
- 课程出勤率（%）
- 考试是否通过（0/1）

## 使用说明

1. 训练区域（左侧）：
   - 调整迭代次数（默认50次）
   - 点击"开始训练"开始模型训练
   - 观察损失函数曲线变化
   - 查看权重参数更新

2. 预测区域（右侧）：
   - 输入学习时间和出勤率
   - 查看预测结果和概率
   - 观察数据点分布

## 技术栈

- React 18
- TypeScript
- D3.js
- ML-Matrix
- Vite 