import os

def remove_info(path):
    for current_dir, dirs, files in os.walk(path):
        for file in files:
            if file == "info.csv":
                os.remove(os.path.join(current_dir, file))

if __name__ == '__main__':
    remove_info("./././Data/test_data/mid_presentation/hoshinogen")