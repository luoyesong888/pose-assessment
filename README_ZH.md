# KinetiQ

KinetiQ 是一个基于 Streamlit 的 AI 运动康复筛查应用。它使用 MediaPipe 对正面和侧面姿态照片进行分析，推断 ACL 风险、动力链问题和可能的肌群功能异常，再交给 DeepSeek 生成面向治疗师的专业报告。

## 功能

- 正面与侧面照片上传
- 姿态标记图生成
- ACL 风险分层
- 动力链分析
- 肌群功能推断
- DeepSeek 个性化报告
- 患者档案与历史记录
- Markdown、HTML、PDF 导出

## 为什么值得展示

- 不是单纯的人体关键点展示，而是把结果翻译成临床可读内容。
- 面向治疗师，而不是普通用户截图式展示。
- 支持患者档案，方便随访对比。
- 没有 API Key 时也能自动回退到本地报告模板。

## 流程

```mermaid
flowchart LR
    A[上传正面与侧面照片] --> B[MediaPipe 关键点识别]
    B --> C[姿态指标与 ACL 风险]
    C --> D[肌群功能推断]
    D --> E[DeepSeek 生成报告]
    E --> F[档案存储 + PDF/HTML 导出]
```

## 快速开始

1. 创建并激活虚拟环境。
2. 安装依赖。
3. 将 `pose_landmarker.task` 放在项目根目录。
4. 如果要使用 DeepSeek 生成报告，设置 `DEEPSEEK_API_KEY`。
5. 启动应用：

```bash
streamlit run main.py
```

## 项目结构

- `main.py`：Streamlit 前端界面
- `app_pipeline.py`：完整评估流程
- `pose.py`：MediaPipe 姿态识别和标记图片生成
- `analysis.py`：姿态指标、ACL 风险、动力链总结
- `clinical_knowledge.py`：肌群映射和报告模板
- `deepseek_client.py`：DeepSeek API 调用
- `records_store.py`：患者档案存储
- `report_export.py`：HTML 和 PDF 导出

## 说明

- 患者记录和上传图片会保存在本地 `data/` 目录。
- 如果没有设置 `DEEPSEEK_API_KEY`，程序会自动使用本地报告模板。
- 这个项目用于筛查和工作流辅助，不是医学诊断工具。
