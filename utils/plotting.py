import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def boxplot(data, feature, target=None, figsize=(7, 4), dpi=100, fname=None):
    """
    Plot univariate or bivariate boxplot of input data.
    """
    plt.figure(figsize=figsize, dpi=dpi)

    if target is None:
        sns.boxplot(data, x=feature)
        var='Univariate'
    else:
        sns.boxplot(data, x=feature, hue=target)
        var='Bivariate'
    
    plt.title("{} boxplot of '{}'".format(var, feature), fontweight='bold')

    plt.tight_layout()

    # Save figure
    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches='tight')

    plt.show()


def histogram(data, feature, target=None, figsize=(7, 4), dpi=100, bins=10, kde=False, fname=None):
    """
    Plot univariate or bivariate histogram of input data.
    """
    plt.figure(figsize=figsize, dpi=dpi)

    if target is None:
        sns.histplot(data[feature], stat='density', bins=bins, kde=kde)
        var='Univariate'
    else:
        sns.histplot(data, x=feature, hue=target, stat='density', bins=bins, common_norm=False, kde=kde)
        var='Bivariate'

    plt.title("{} histogram of '{}'".format(var, feature), fontweight='bold')

    plt.tight_layout()

    # Save figure
    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches='tight')

    plt.show()


def sdg_histogram(data1, data2, feature1, feature2, figsize=(10, 5), dpi=100, bins=10, kde=False, fname=None):
    """
    Plot bivariate histogram of input data.
    """
    plt.figure(figsize=figsize, dpi=dpi)

    sns.histplot(data1, x=feature1, stat='density', bins=bins, common_norm=False, kde=kde, label='Original')
    sns.histplot(data2, x=feature2, stat='density', bins=bins, common_norm=False, kde=kde, label='Synthetic')

    plt.title("Original vs Synthetic histogram of '{}'".format(feature1), fontweight='bold')

    plt.tight_layout()
    plt.legend()

    # Save figure
    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches='tight')

    plt.show()


def sdg_boxplot(data1, data2, feature1, feature2, figsize=(10, 5), dpi=100, fname=None):
    """
    Plot bivariate boxplot of input data.
    """
    plt.figure(figsize=figsize, dpi=dpi)
    
    feature = feature1
    df1 = data1[[feature]].copy()
    df1[['']] = 'Original'
    df2 = data2[[feature2]].copy()
    df2.rename(columns={feature2: feature})
    df2[['']] = 'Synthetic'
    data = pd.concat([df1, df2], axis=0)

    sns.boxplot(data, x=feature, hue='', gap=.1, legend='full')
 
    # sns.boxplot(data1, x=feature1, ax=ax[0], label='Original')
    # sns.boxplot(data2, x=feature2, ax=ax[1], label='Synthetic')

    plt.title("Original vs Synthetic boxplot of '{}'".format(feature1), fontweight='bold')

    plt.tight_layout()

    # Save figure
    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches='tight')

    plt.show()
    

def corr_heatmap(data, figsize=(10, 10), dpi=100, mask=False, annot=False, cbar=False, fname=None):
    """
    Plot correlation heatmap of input data.
    """
    corr=data.corr()

    plt.figure(figsize=figsize, dpi=dpi)
    
    # Use mask to plot heatmap as a lower triangular matrix
    if mask:
        mask=np.triu(np.ones_like(corr, dtype=np.bool_))
    
    # Define heatmap object
    heatmap = sns.heatmap(
        corr
      , mask=mask
      , vmin=-1.
      , vmax=1.
      , cmap='coolwarm'
      , center=False
      , annot=annot
      , cbar=cbar
    )

    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 30}, pad=20)

    plt.tight_layout()
    
    # Save figure
    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches='tight')

    plt.show()


def target_corr_heatmap(data, target, figsize=(10, 10), dpi=100, annot=False, cbar=False, fname=None):
    """
    Plot correlation heatmap of features with target.
    """
    corr=data.corr()[[target]].iloc[:-1].sort_values(by=target, ascending=False)
    
    # Define figure
    plt.figure(figsize=figsize, dpi=dpi)

    # Define heatmap object
    heatmap = sns.heatmap(
        corr
      , vmin=-1.
      , vmax=1.
      , annot=annot
      , cmap='coolwarm'
      , cbar=cbar
    )

    heatmap.set_title('Features correlation with target', fontdict={'fontsize': 18}, pad=16)

    plt.tight_layout()

    # Save figure
    if fname is not None:
        plt.savefig(fname, dpi=300, bbox_inches='tight')

    plt.show()
    