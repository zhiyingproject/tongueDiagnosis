import pathlib
import shutil
import pandas as pd


def organize_f(original_p, to_p, config_f):

    labels = pd.read_csv(config_f)
    categories = labels.columns[1:]
    for category in categories:
        grouped_f = labels.groupby([category])[labels.columns[0]]
        for group in grouped_f:
            for f in group[1]:
                to_p_category = pathlib.Path(f"{to_p}/{category}/{group[0]}")
                to_p_category.mkdir(parents=True, exist_ok=True)
                file_name_a = f'{f}a.jpg'
                from_p_a = pathlib.Path(f'{original_p}/{file_name_a}')
                shutil.copy(from_p_a, to_p_category)

                '''file_name_b = f'{f}b.jpg'
                from_p_b = pathlib.Path(f'{original_p}/{file_name_b}')
                shutil.copy(from_p_b, to_p_category)'''


