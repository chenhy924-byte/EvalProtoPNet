# python get_cpu_con-sta-vis_cmds.py
import re
import os

def generate_commands():
    print("="*60)
    print("请粘贴 Windows 路径 (例如: D:\\SeniorThesis\\...\\2p_resnet34_...)")
    print("="*60)
    
    raw_input = input("> ").strip()
    
    if not raw_input:
        return

    # 1. 提取文件夹全名 (处理 Windows 或 Linux 路径分隔符)
    folder_name = os.path.basename(raw_input.rstrip('\\/'))
    
    # 2. 正则解析: 提取人数 (num) 和 架构 (arch)
    # 匹配格式: {数字}p_{架构}_{日期}...
    match = re.search(r'(\d+)p_([a-zA-Z0-9]+)_', folder_name)
    
    if match:
        p_num = match.group(1)   # 提取出 "2" 或 "5"
        arch = match.group(2)    # 提取出 "resnet34" 或 "densenet161"
    else:
        print("错误：无法从路径中解析出人数和架构，请检查格式是否为 '2p_arch_...'")
        return

    # 3. 构造路径参数
    data_path = f"datasets/Barefoot_Dataset_{p_num}"
    resume_path = f"output_cosine/{folder_name}/checkpoints/save_model.pth"
    
    # 4. 定义模板 (使用 PowerShell 的换行符 ` )
    templates = [
        ("一致性得分", "eval_consistency.py", ""),
        ("稳定性得分", "eval_stability.py", ""),
        ("可视化结果", "local_analysis_vis.py", "  --vis_classes 0 1")
    ]

    print("\n" + "生成结果如下：" + "\n" + "-"*30)

    for label, script, extra in templates:
        cmd = (f"{label}：\n"
               f"python {script} `\n"
               f"  --data_set Barefoot_Dataset `\n"
               f"  --data_path \"{data_path}\" `\n"
               f"  --base_architecture {arch} `\n"
               f"  --resume \"{resume_path}\" `\n"
               f"  --half_size 36{extra}\n")
        print(cmd)

if __name__ == "__main__":
    generate_commands()
    input("\n按回车键退出...")

# python get_cpu_con-sta-vis_cmds.py