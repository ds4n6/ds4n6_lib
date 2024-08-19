import pandas as pd

def explore(df, col, max_rows=None, max_columns=None):
    hist = df[col].value_counts()
    with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_columns):
        print("#Count:",len(hist))
        print(hist)