import os

def read_tsv(tsv_file_path):
    data = {}
    with open(tsv_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                data[os.path.splitext(parts[0])[0]] = parts[1]
    return data


	
def process_images(image_folder, tsv_file_path):
    tsv_data = read_tsv(tsv_file_path)
    
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            file_name_without_ext = os.path.splitext(filename)[0]
            if file_name_without_ext in tsv_data:
                content = tsv_data[file_name_without_ext]
                output_file_path = os.path.join(image_folder, f"{file_name_without_ext}.txt")
                with open(output_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(content)
                print(f"Created file: {output_file_path}")
            else:
                print(f"No entry found for {file_name_without_ext} in TSV file.")

if __name__ == "__main__":

    image_folder_path = "./datasets/test"
    tsv_file_path = "./datasets/labels.tsv"
    
    process_images(image_folder_path, tsv_file_path)
