import statsmodels.api as sm
import statsmodels.stats.outliers_influence as oif
import scipy.stats as sc_ss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def plot_odds(field, df, label_col: str='label',
                 bins: int=10, apply_func: bool=False,
                 y_field: str='log_odds'):
    """Checks linear assumption between input features and logodds of outcome"""

    sub = df[[field, label_col]].copy()

    if apply_func:
        #TO DO: Update function to be based on input parameters
        sub[field] = StandardScaler().fit_transform(np.array(sub[field]).reshape(-1,1))
        sub[field] = [1/np.exp(x) for x in sub[field]]
    
    x_field = f'{field}_bins'
    sub[x_field] = pd.cut(sub[field], bins)
    
    sub_grouped = sub.groupby(x_field).agg({label_col : [sum, len]})
    sub_grouped.columns = ['total_class', 'n']
    sub_grouped['p'] = sub_grouped.total_class / sub_grouped.n
    sub_grouped['log_odds'] = [math.log1p(p / (1-p)) for p in sub_grouped.p]
    sub_grouped.reset_index(inplace=True)
    
    y_field_map = {'log_odds' : 'Log Odds',
                   'p' : 'Probability',
                   'total_class': 'Count Class',
                   'n' : 'Total Count'}
    y_label = y_field_map[y_field]
    
    plt.style.use('seaborn')
    fig, ax = plt.subplots()
    ax.scatter(sub_grouped[x_field].astype(str), sub_grouped[y_field])
    for tick in ax.get_xticklabels():
        tick.set_rotation(75)
    ax.set_xlabel(f'{field} Bins')
    ax.set_ylabel(y_label)
    ax.set_title(f"Checking Linearity of {y_label} and {field}")
    return sub_grouped

def check_correlations(input_df: pd.DataFrame, plot_size: int=10):
    """Automatically iterates through and drops columns with detected multicollinearity"""

    corr = input_df.corr()
    fig, ax = plt.subplots(figsize=(plot_size, plot_size))
    ax.matshow(corr, cmap=plt.cm.RdYlGn_r)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.xticks(range(input_df.select_dtypes(['number']).shape[1]), input_df.select_dtypes(['number']).columns, fontsize=14, rotation=90)
    plt.yticks(range(input_df.select_dtypes(['number']).shape[1]), input_df.select_dtypes(['number']).columns, fontsize=14)
    plt.tight_layout()
    plt.title("Correlations Matrix")


    vif = oif.variance_inflation_factor
    vifs = {}
    for idx, col in enumerate(input_df.columns):
        vif_val = vif(input_df, idx)
        vifs[col] = vif_val
    
    vifs = pd.Series(vifs)
    print(vifs)
    return vifs

def adjust_vif(input_df: pd.DataFrame):
    """Automatically iterates through and drops columns with detected multicollinearity"""

    thresh = 10
    init_vif = check_correlations(input_df)
    max_vif = max(init_vif)
    drop_cols = init_vif[init_vif == max_vif].index.to_list()
    while max_vif > thresh:
        new_df = input_df.drop(drop_cols, axis=1)
        vif_test = check_correlations(new_df)
        max_vif = max(vif_test)
        if max_vif < thresh:
            break
        drop_col = vif_test[vif_test == max_vif].index.to_list()
        drop_cols.extend(drop_col)
    
    return drop_cols

def logit_comparison(df_reduced, df_full, y, thresh: float=0.05):
    """Compares Full Model with Reduced Model for Binomial GLM and returns
    right tail probability of a Chi-Squared distribution. Prob <= threshold
    leads to rejecting H0, and full model has more explanatory power than reduced"""

    reduced_model = sm.GLM(y, df_reduced, family=sm.families.Binomial()).fit()
    full_model = sm.GLM(y, df_full, family=sm.families.Binomial()).fit()
    D = reduced_model.deviance - full_model.deviance

    right_tail_prob = sc_ss.chi2.sf(D, len(X_sm.columns) - len(X_base.columns))
    if right_tail_prob <= thresh:
        print("Full Model appears to have more explanatory Power")
    else:
        print("Full Model does not appear to have more explanatory power")
    return right_tail_prob

def logit_fit_test(logit_model: sm.GLM, thresh: float=0.05, bins: int=100):
    """Generates plots of logit residuals, and runs test on whether
    residuals are approximately normal, and outputs results based on 
    H0 that residuals are normally distributed based on Chi-Squared test"""

    ri = np.array(logit_model.resid_pearson)
    fig, ax = plt.subplots()
    ax.hist(ri, bins=bins)
    ax.set_title("Pearson Residual Distribution")
    ri = ri.dot(ri)

    di = np.array(logit_model.resid_deviance)
    fig, ax = plt.subplots()
    ax.hist(di, bins=bins)
    ax.set_title("Deviance Residual Distribution")
    di = di.dot(di)
    
    pearson_prob = sc_ss.chi2.sf(ri, len(X_sm) - len(X_sm.columns) - 1)
    dev_prob = sc_ss.chi2.sf(di, len(X_sm) - len(X_sm.columns) - 1)
    
    print(f'Pearson Prob: {pearson_prob}')
    print(f'Deviance Prob: {dev_prob}')
    if (pearson_prob >= thresh and dev_prob >= thresh):
        print("Model appears to have Good Fit")
    elif (pearson_prob <= thresh and dev_prob <= thresh):
        print("Model does not appear to have Good Fit")
    else:
        print("Mixed Results")
    return pearson_prob, dev_prob
