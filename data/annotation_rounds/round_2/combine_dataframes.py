import pandas as pd


def main():
    r1 = pd.read_csv("R1_Universal_Annotation_Results.csv").iloc[:, 1:]
    r2 = pd.read_csv("R2_Universal_Annotation_Results.csv").iloc[:, 1:]

    df = pd.concat([r1, r2])
    df = df.sort_values(by='FILENAME')
    df = df.reset_index(drop=True)

    df.to_csv("Universal_Annotation_Results.csv")


if __name__ == "__main__":
    main()
