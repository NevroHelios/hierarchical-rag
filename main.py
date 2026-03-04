import os
from unstructured.partition.text import partition_text
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import elements_to_json, elements_to_md
# from unstructured.cleaners.core import clean_extra_whitespace

filename = "abstracts1.txt"
path = "data/fragmented"
output_path = "data/processed"

def extract_to_json(filename: str):
    src = os.path.join(path, filename)
    contents = partition_text(src)
    elements_to_json(contents, os.path.join(output_path, f"{filename.split('.')[0]}.json"))

def extract_to_readme(filename: str):
    src = os.path.join(path, filename)
    contents = partition_pdf(filename=src,
                            languages=['en'],
                            strategy='hi_res',
                            infer_table_structure=True,
                            hi_res_model_name='yolox' 
                            )
    elements_to_md(contents, os.path.join(output_path, f"{filename.split('.')[0]}.md"))




if __name__ == '__main__':
    # extract_to_json(filename)
    # filename = "clinical_reports.txt"
    # path = "data/processed"
    # extract_to_json(filename="book1.md")
    # filename = "book1.pdf"
    # path = "data"
    # extract_to_readme(filename)
    ...