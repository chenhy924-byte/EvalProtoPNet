import os


def count_images_in_dataset():
    # 定义常见的图片后缀
    valid_extensions = ('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff')

    while True:
        print("\n" + "=" * 60)
        root_path = input("请输入数据集根目录路径 (例如 Barefoot_Dataset，按 Q 退出): ").strip()

        if root_path.upper() == 'Q':
            break

        if not os.path.exists(root_path):
            print(f"❌ 路径不存在: {root_path}")
            continue

        # 获取根目录下所有的子文件夹 (test_cropped, train_cropped 等)
        try:
            main_folders = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

            if not main_folders:
                print("📝 该目录下没有子文件夹。")
                continue

            print(f"✅ 找到 {len(main_folders)} 个主要文件夹: {', '.join(main_folders)}")

            for main_folder in sorted(main_folders):
                current_main_path = os.path.join(root_path, main_folder)

                print("\n" + "-" * 20)
                print(f"📂 正在统计文件夹: 【{main_folder}】")
                print("-" * 20)

                total_images_in_main = 0
                subfolder_stats = {}

                # 遍历子文件夹 (如 000, 001, 002...)
                items = os.listdir(current_main_path)
                for item in items:
                    item_path = os.path.join(current_main_path, item)

                    if os.path.isdir(item_path):
                        # 统计每个类别文件夹里的图片
                        count = len([f for f in os.listdir(item_path)
                                     if f.lower().endswith(valid_extensions)])
                        subfolder_stats[item] = count
                        total_images_in_main += count
                    elif item.lower().endswith(valid_extensions):
                        # 如果主文件夹下直接有图片
                        total_images_in_main += 1

                # 打印当前主文件夹的详细结果
                if subfolder_stats:
                    # 为了不让屏幕滚太长，这里可以只列出前几个和后几个，或者全部列出
                    # 这里依然保持全部列出，你可以根据需要修改
                    for sub_dir, count in sorted(subfolder_stats.items()):
                        print(f"  sub-dir -> {sub_dir}: {count} 张")

                print(f"\n✨ 【{main_folder}】 统计完成:")
                print(f"   - 子文件夹(类别)总数: {len(subfolder_stats)}")
                print(f"   - 图片总数量: {total_images_in_main}")

            print("\n" + "=" * 60)
            print("🎉 所有文件夹扫描完毕！")

        except Exception as e:
            print(f"⚠️ 发生错误: {e}")


if __name__ == "__main__":
    count_images_in_dataset()
