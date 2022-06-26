import numpy
import pandas as pd
from scipy.stats import kendalltau


def get_annotation_df(filename):
    """
    Get limited annotation DataFrame

    :param filename: String representing dataset filename
    :return: DataFrame with first and last 20 entries
    """
    df = pd.read_csv(filename).sort_values(by=['Probability.Narrative'])
    annotation_df = pd.concat([df.head(20), df.tail(20)])
    return annotation_df


def print_divider():
    """
    Print divider line
    """
    print(50 * '-')
    pass


def start_annotation(df):
    """
    Small command line interface for annotation
    :param df: DataFrame for annotation
    :return: DataFrame after annotation
    """
    # Shuffle and insert new columns
    df = df.sample(frac=1).reset_index(drop=True)
    universals = ['suspense', 'surprise', 'curiosity']
    for universal in universals:
        df.insert(5, universal, [0] * len(df))
    df.insert(5, 'universal_avg', [0] * len(df))

    # Start annotation
    print('You will be presented with a random passage. Rate this passage on {0}, {1} and {2}.'.format(universals[0],
                                                                                                       universals[1],
                                                                                                       universals[2]))
    for index, row in df.iterrows():
        print_divider()
        print(row['TEXT'])
        print()

        annotation_list = []
        for universal in universals:
            val = int(input('On a scale from 1 to 5, I rate \'{0}\': '.format(universal)))
            annotation_list.append(val)
            df.loc[df['FILENAME'] == row['FILENAME'], universal] = val
        df.loc[df['FILENAME'] == row['FILENAME'], 'universal_avg'] = sum(annotation_list) / len(annotation_list)

    return df


def main():
    numpy.random.seed(2022)

    annotation_done = True
    original_file = 'MinNarrative_ReaderData_Final.csv'
    annotation_file = 'MinNarrative_ReaderData_Universals.csv'

    if not annotation_done:
        df = get_annotation_df(original_file)
        df = start_annotation(df)
        df.to_csv(annotation_file, index=False)
    else:
        df = pd.read_csv(annotation_file)

        annotators = ['AY', 'ML', 'LM']
        for annotator in annotators:
            df['{0}_avg'.format(annotator)] = (df['{0}_agency'.format(annotator)].values +
                                               df['{0}_event'.format(annotator)].values +
                                               df['{0}_world'.format(annotator)].values) / 3
        narrativity = list(df.sort_values(by=['Probability.Narrative'])['FILENAME'].values)
        universal_avg = list(df.sort_values(by=['universal_avg'])['FILENAME'].values)
        avg = list(df.sort_values(by=['avg_overall'])['FILENAME'].values)

        print('Correlation Universals & Narrative Probability:    {0}'.format(kendalltau(narrativity,
                                                                                         universal_avg).correlation))
        print('Correlation Piper (2022) & Narrative Probability:  {0}'.format(kendalltau(narrativity,
                                                                                         avg).correlation))


if __name__ == "__main__":
    main()
