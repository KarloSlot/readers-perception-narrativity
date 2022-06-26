import pandas as pd
import numpy as np


def get_ADI(df_numpy):
    avg_deviation_list = []
    for row in df_numpy:
        deviation_list = []
        for i in range(3):
            aspect_list = np.array([row[x + (i * 3)] for x in range(3)])
            deviation_list.append(aspect_list.mean())
        deviation_list = np.array(deviation_list)
        avg_deviation_list.append(sum(np.abs(deviation_list - deviation_list.mean())) / 3)
    return round(np.array(avg_deviation_list).mean(), 3)


def main():
    col_uni = ['{0}_cmb_{1}'.format(statement, annotator) for annotator in range(3) for statement in
               ['suspense', 'curiosity', 'surprise']]
    df_uni = pd.read_csv('R1_Universal_Annotation_Results.csv')
    df_uni_val_1 = df_uni[col_uni].to_numpy()
    files_uni = list(df_uni['FILENAME'].values)

    df_uni = pd.read_csv('R2_Universal_Annotation_Results.csv')
    df_uni_val_2 = df_uni[col_uni].to_numpy()

    df_uni = pd.read_csv('Universal_Annotation_Results.csv')
    df_uni_val = df_uni[col_uni].to_numpy()

    df_p22 = pd.read_csv('MinNarrative_ReaderData_Final.csv')
    df_p22 = df_p22.loc[df_p22['FILENAME'].isin(files_uni)]
    col_p22 = ['{0}_{1}'.format(annotator, statement) for annotator in ['AY', 'ML', 'LM'] for statement in
               ['agency', 'event', 'world']]
    df_p22_val = df_p22[col_p22].to_numpy()

    df_p21 = pd.read_csv('annotated_dataset_401.csv')
    col_p21 = ['{0}_{1}'.format(annotator, statement) for annotator in ['C1', 'C2', 'C3'] for statement in
               ['agency', 'event', 'world']]
    df_p21_val = df_p21[col_p21].to_numpy()[:-1]

    print('ADI - Piper (2021): ', get_ADI(df_p21_val))
    print('ADI - Piper (2022): ', get_ADI(df_p22_val))
    print('ADI - Universals R1: ', get_ADI(df_uni_val_1))
    print('ADI - Universals R2: ', get_ADI(df_uni_val_2))
    print('ADI - Universals: ', get_ADI(df_uni_val))


if __name__ == "__main__":
    main()
