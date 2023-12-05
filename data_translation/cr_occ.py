import pandas as pd


def filter_df_rows(df, col, val_list):
    filtered_df = df[df[col].isin(val_list)]
    return filtered_df


def filter_df_cols(df, col_list):
    filtered_df = df[col_list]
    return filtered_df


def load_raw_occurrence(csv_file):
    '''Sorts csv for only common reed data in New York'''
    df = pd.read_csv(csv_file)
    rowf1_df = filter_df_rows(df, "scientific_name", [
                              'Phragmites australis ssp. australis', 'Phragmites australis ssp. Australis', 'Phragmites australis'])
    rowf2_df = filter_df_rows(rowf1_df, "jurisdiction", ['New York'])
    crf_df = filter_df_cols(rowf2_df, ['observation_date', 'number_found'])
    return crf_df


def get_timeseries_data(df):
    ''' makes cumulative sum of common reeds over time. Note: change in date format (metadata)'''
    df['observation_date'] = pd.to_datetime(
        df['observation_date'], format='%d-%b-%Y')
    df.sort_values(by='observation_date', inplace=True)
    df['running_total'] = df['number_found'].cumsum()
    return df


if __name__ == "__main__":
    filt_df = load_raw_occurrence('raw_occurrence_data.csv')
    ts_df = get_timeseries_data(filt_df)
    ts_df.to_csv('common_reed_occurrence_timeseries.csv', index=False)
