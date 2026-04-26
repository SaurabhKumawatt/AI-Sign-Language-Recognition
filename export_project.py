import os

# Automatically take current folder
PROJECT_DIR = os.getcwd()
OUTPUT_FILE = "project_dump.txt"

# File extensions to include
CODE_EXTENSIONS = [".py", ".js", ".html", ".css", ".json", ".txt"]

# Folders to ignore
IGNORE_FOLDERS = ["venv", "node_modules", "__pycache__", ".git"]

def write_structure_and_code():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for root, dirs, files in os.walk(PROJECT_DIR):
            
            # Skip unwanted folders
            dirs[:] = [d for d in dirs if d not in IGNORE_FOLDERS]

            level = root.replace(PROJECT_DIR, '').count(os.sep)
            indent = ' ' * 4 * level
            out.write(f"{indent}📁 {os.path.basename(root)}\n")

            sub_indent = ' ' * 4 * (level + 1)

            for file in files:
                out.write(f"{sub_indent}📄 {file}\n")

                if any(file.endswith(ext) for ext in CODE_EXTENSIONS):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()

                        out.write(f"{sub_indent}--- START OF {file} ---\n")
                        out.write(content + "\n")
                        out.write(f"{sub_indent}--- END OF {file} ---\n\n")

                    except Exception as e:
                        out.write(f"{sub_indent}[Error reading file: {e}]\n\n")

if __name__ == "__main__":
    write_structure_and_code()
    print("✅ Done! project_dump.txt created in same folder")