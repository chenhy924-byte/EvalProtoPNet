# python get_gpu_con-sta-vis_cmds.py

import re

def _ask(prompt: str, default: str) -> str:
    s = input(f"{prompt} (默认: {default})\n> ").strip()
    return s if s else default


def generate_commands():
    print("请输入复制的路径 (例如: autodl-tmp/EvalProtoPNet/output_cosine/...)")
    raw_input = input("> ").strip()
    
    if not raw_input:
        print("输入为空，请重新运行。")
        return

    # 1. 提取存储位置 (tmp 或 fs)
    storage_type = "autodl-tmp" if "autodl-tmp" in raw_input else "autodl-fs"
    
    # 2. 提取实验文件夹名 (最后一级)
    folder_name = raw_input.split('/')[-1]
    
    # 3. 解析人数和架构: {数字}p_{arch}_{date}...
    match = re.search(r'(\d+)p_([a-zA-Z0-9]+)_', folder_name)
    if match:
        p_num = match.group(1)
        arch = match.group(2)
    else:
        p_num = "200"
        try:
            arch = folder_name.split('_')[1]
        except IndexError:
            arch = "unknown"

    # 模板配置
    default_data_root = f"/root/{storage_type}/datasets"
    data_root = _ask("请输入数据集根目录（例如: /root/autodl-tmp/datasets）", default_data_root)
    data_path = f"{data_root}/Barefoot_Dataset_{p_num}"
    # main.py now saves best model as best_model.pth and final model as final_model.pth
    resume_path = f"output_cosine/{folder_name}/checkpoints/best_model.pth"
    
    templates = [
        ("一致性得分", "eval_consistency.py"),
        ("稳定性得分", "eval_stability.py"),
        ("可视化结果", "local_analysis_vis.py")
    ]

    print("\n" + "="*50)
    print(f"检测到人数: {p_num} | 架构: {arch} | 存储位置: {storage_type}")
    print("="*50 + "\n")

    for label, script in templates:
        cmd = (f"{label}：\n"
               f"python {script} --data_set Barefoot_Dataset "
               f"--data_path {data_path} "
               f"--base_architecture {arch} "
               f"--resume {resume_path} --half_size 36\n")
        print(cmd)

if __name__ == "__main__":
    generate_commands()

# python get_gpu_con-sta-vis_cmds.py