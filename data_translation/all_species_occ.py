from cr_occ import filter_df_rows, filter_df_cols
import pandas as pd


def load_ny_occurrence(csv_file):
    '''Sorts presence_line.csv for necessary New York data'''
    df = pd.read_csv(csv_file)
    rowf1_df = filter_df_rows(df,"jurisdiction",['New York'])
    crf_df = filter_df_cols(rowf1_df,['observation_date','number_found', 'common_name'])
    return crf_df

def get_spec_timeseries_data(df):
    ''' makes cumulative sum of common reeds over time. Note: change in date format (metadata)'''
    df['observation_date'] = pd.to_datetime(df['observation_date'], format='%d-%b-%Y')
    df.sort_values(by='observation_date', inplace=True)
    df['running_total_per_species'] = df.groupby('common_name')['number_found'].cumsum()
    df['running_total_all'] = df['number_found'].cumsum()
    return df

def main():
    filt_df = load_ny_occurrence('raw_occurrence_data.csv')
    ts_df = get_spec_timeseries_data(filt_df)
    ts_df.to_csv('all_species_occurrence_timeseries.csv',index=False)


if __name__ == "__main__":
    main()