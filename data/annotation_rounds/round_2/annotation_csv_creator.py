import pandas as pd
import numpy as np
import random
import os


def construct_df():
    annotator_count = len([name for name in os.listdir('.') if os.path.isfile(name)
                           and name.endswith('_universals_R2.xlsx')])
    universals = ['suspense', 'curiosity', 'surprise']

    files = []
    df_universals = pd.DataFrame()
    for i in range(annotator_count):
        if i >= 4:
            i += 1
        sheet_name = '{0}_universals_R2.xlsx'.format(i)
        df = pd.read_excel(sheet_name)
        df = df.iloc[:, 1:-1]
        df = df.rename(columns={universal: '{0}_ann_{1}'.format(universal, i) for universal in universals})
        df_universals = pd.concat([df_universals, df])
        files += list(df['FILENAME'])
    files = sorted(list(set(files)))

    df_universals_new = pd.DataFrame()
    for file in files:
        temp_df = df_universals.loc[df_universals['FILENAME'] == file].max().to_frame().T
        df_universals_new = pd.concat([df_universals_new, temp_df])
    df_universals_new = df_universals_new.reset_index(drop=True)

    for index, row in df_universals_new.iterrows():
        values = [value for value in row[2:].values if value > 0]
        counter = 0
        for i in range(3):
            for universal in universals:
                df_universals_new.loc[index, '{0}_cmb_{1}'.format(universal, i)] = values[counter]
                counter += 1

    df_universals_new['avg_universal'] = df_universals_new.iloc[:, 23:].mean(numeric_only=True, axis=1)
    return df_universals_new


def calculate_irr(df):
    print(df)


def main():
    seed = 1818
    np.random.seed(seed)
    random.seed(seed)

    df_universals = construct_df()
    df_universals.to_csv('R2_Universal_Annotation_Results.csv')


if __name__ == "__main__":
    main()
