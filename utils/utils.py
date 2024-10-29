import pandas as pd
from scipy import stats

def normalize(df):
    """Normalizes input Pandas DataFrame."""
    return (df - df.min()) / (df.max() - df.min())


def inv_normalize(norm_df, min_max_df):
    """Inverts normalization of input Pandas DataFrame."""
    df = norm_df.copy()
    for col in norm_df.columns:
        min = min_max_df[col].loc['min']
        max = min_max_df[col].loc['max']
        df[col] = norm_df[col] * (max - min) + min
    return df


def melt_corr(frame):
    """
    Computes the correlation matrix of the input dataframe and returns 
    a non-redundant melted dataframe with three columns:
    - feature 1,
    - feature 2,
    - correlation coefficient.
    One entry for each pair of features of the input dataframe. 
    """
    # Melt the input dataframe to
    corr_frame = frame.corr()
    corr_frame.reset_index(names='feat_2', inplace=True)
    corr_frame = pd.melt(corr_frame, id_vars='feat_2', var_name='feat_1', value_name='corr_coeff')
    corr_frame = corr_frame[['feat_1', 'feat_2', 'corr_coeff']]

    # For each value of feat_1, find the index of the row where feat_1 and feat_2 are equal
    corr_frame['cut'] = corr_frame.apply(lambda x: 1 if x['feat_1']==x['feat_2'] else 0, axis=1)
    corr_frame = pd.merge(
        corr_frame
      , corr_frame.groupby('feat_1').apply(lambda x: x['cut'].idxmax()).to_frame(name='idxmax')
      , how='left'
      , on='feat_1'
    )

    # Drop redundant entries
    corr_frame.reset_index(inplace=True)
    corr_frame = corr_frame.loc[corr_frame['index']>corr_frame['idxmax']]
    corr_frame.reset_index(drop=True, inplace=True)

    # Drop auxiliary columns
    corr_frame.drop(columns=['index', 'cut', 'idxmax'], inplace=True)

    return corr_frame


def corr_compare(orig, synth):
    """
    """
    # Check if the input DataFrames have the same columns in the same order
    if not orig.columns.tolist() == synth.columns.tolist():
        raise ValueError('Input DataFrames must have the same column names in the same order.')
    corr_orig = melt_corr(orig)
    corr_synth = melt_corr(synth)
    corr = pd.concat([corr_orig, corr_synth[['corr_coeff']]], axis=1)
    return corr


def normality_tests(frame, norm=False):
    """
    Performs Anderson, Kolmogorov-Smirnov and Shapiro-Wilk normality 
    tests on input dataframe. Normalizes data before performing the tests 
    if the 'norm' parameter is set to True. The p-value is set to 0.05 
    by default.

    Returns a dataframe with the same columns as the input dataframe 
    and one row for each test. A cell reads True if the null hypothesis
    is rejected, which means that there is evidence that data follow 
    a non-normal distribution. Otherwise, False indicates that the null 
    hypothesis cannot be rejected, so data are normally distributed.
    """
    tests_frame = pd.DataFrame()

    # Normalize input dataframe
    if norm:
        frame = normalize(frame)

    # Anderson
    tests_lambda = lambda x: (lambda t: t.statistic<t.critical_values[2])(stats.anderson(x, dist='norm'))
    tests_frame = pd.concat([tests_frame, frame.apply(tests_lambda, axis=0).to_frame(name='anderson-darling')], axis=1)

    # Kolmogorov-Smirnov
    tests_lambda = lambda x: (lambda t: t.pvalue>.05)(stats.kstest(x, stats.norm.cdf))
    tests_frame = pd.concat([tests_frame, frame.apply(tests_lambda, axis=0).to_frame(name='kolmogorov-smirnov')], axis=1)

    # Shapiro-Wilk
    tests_lambda = lambda x: (lambda t: t.pvalue>.05)(stats.shapiro(x))
    tests_frame = pd.concat([tests_frame, frame.apply(tests_lambda, axis=0).to_frame(name='shapiro-wilk')], axis=1)
    
    return tests_frame.T
