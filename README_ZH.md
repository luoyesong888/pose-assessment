# KinetiQ

KinetiQ 是一个基于 Streamlit 的 AI 运动康复筛查应用。它使用 MediaPipe 自动识别正面、背面、侧面、前屈和其他姿态照片，按照片类型进行描述性分析，整理可见对线、动作对称性、动力链观察和肌群功能假设，再交给 DeepSeek 生成面向治疗师的专业报告。

## 功能

- 任意角度与动作姿态照片上传
- 姿态标记图生成
- ACL 筛查边界提示（静态照片不输出风险等级）
- 动力链分析
- 肌群功能推断
- DeepSeek 个性化报告
- 患者档案与历史记录
- 本地多视角姿态 RAG：正/背面、侧面、前屈和其他姿态分类检索 MPII 关键点相似案例与证据边界
- 分析后改善重点推荐、用户确认与低负荷训练计划
- Markdown、HTML、PDF 导出

## 为什么值得展示

- 不是单纯的人体关键点展示，而是把结果翻译成临床可读内容。
- 面向治疗师，而不是普通用户截图式展示。
- 支持患者档案，方便随访对比。
- 没有 API Key 时也能自动回退到本地报告模板。

## 流程

```mermaid
flowchart LR
    A[上传任意角度体态照片] --> B[MediaPipe 关键点 + 照片类型识别]
    B --> C[静态姿态指标与质量门禁]
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
- `analysis.py`：姿态指标、照片质量门禁、动力链总结与 ACL 筛查边界
- `clinical_knowledge.py`：肌群映射和报告模板
- `deepseek_client.py`：DeepSeek API 调用
- `records_store.py`：患者档案存储
- `report_export.py`：HTML 和 PDF 导出
- `posture_rag.py`：本地向量检索与姿态结构相似度检索
- `recommendation_engine.py`：可解释的垂直改善建议与确认计划
- `scripts/sync_modelscope_posture.py`：同步魔搭体态相关数据
- `scripts/build_posture_rag.py`：蒸馏数据并重建 RAG 索引

## 说明

- 患者记录和上传图片会保存在本地 `data/` 目录。
- 如果没有设置 `DEEPSEEK_API_KEY`，程序会自动使用本地报告模板。
- 这个项目用于筛查和工作流辅助，不是医学诊断工具。
- 更新本地 RAG：`.venv/bin/python scripts/build_posture_rag.py`
