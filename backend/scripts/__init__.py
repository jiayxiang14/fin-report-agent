"""离线评测/工具脚本。做成包是为了让mypy把这里的文件统一识别成scripts.xxx——
eval_trajectory.py 会从同目录导入其它脚本，没有这个文件时mypy会把同一个文件
同时识别成"eval_report_rubric"和"scripts.eval_report_rubric"两个模块而报错（CI挂过）。"""
