import os
import json

gdc_12_canh_dieu_path = r"qa_pairs_ocr/Giao_duc_cong_dan_12_canh_dieu/Giao_duc_cong_dan_12_canh_dieu_qa_pairs.json"
gdc_12_path = r"qa_pairs_ocr/Giao_duc_cong_dan_12/Giao_duc_cong_dan_12_qa_pairs.json"
output_folder = r"qa_pairs_ocr/Giao_duc_cong_dan_12_qa_pairs_combined"
output_file = os.path.join(output_folder, "Giao_duc_cong_dan_12_qa_pairs_combined.json")

os.makedirs(output_folder, exist_ok=True)

with open(gdc_12_canh_dieu_path, 'r', encoding='utf-8') as file:
    gdc_12_canh_dieu_data = json.load(file)

with open(gdc_12_path, 'r', encoding='utf-8') as file:
    gdc_12_data = json.load(file)

max_id = max(int(item["id"].split('_')[-1]) for item in gdc_12_data)
for idx, item in enumerate(gdc_12_canh_dieu_data, start=max_id + 1):
    parts = item["id"].split('_')
    parts[-1] = str(idx)
    item["id"] = '_'.join(parts)

combined_data = gdc_12_data + gdc_12_canh_dieu_data

with open(output_file, 'w', encoding='utf-8') as file:
    json.dump(combined_data, file, ensure_ascii=False, indent=4)

print(f"Combined data with updated IDs has been written to {output_file}")
