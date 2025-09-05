import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3

df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():

    # 5
    df_cat = pd.melt(
    df,
    id_vars=['cardio'],
    value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6
    df_cat = df_cat.value_counts().reset_index(name='total')
    df_cat = df_cat.rename(columns={'variable': 'feature'})

    # 7
    
    g = sns.catplot(
    data=df_cat,
    x="feature", y="total",
    hue="value", col="cardio",
    kind="bar",
    order=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    )


    g.set_axis_labels("variable", "total")
    g.set_titles("cardio = {col_name}")

    # 8
    fig = g.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():

    # 11
    df_heat = df.copy()

    df_heat = df_heat[
        (df_heat['ap_lo'] <= df_heat['ap_hi']) &
        (df_heat['height'] >= df_heat['height'].quantile(0.025)) &
        (df_heat['height'] <= df_heat['height'].quantile(0.975)) &
        (df_heat['weight'] >= df_heat['weight'].quantile(0.025)) &
        (df_heat['weight'] <= df_heat['weight'].quantile(0.975))
    ]

    # 12
    corr = df_heat.corr(numeric_only=True)

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(12, 8))

    # 15

    sns.heatmap(
    corr,
    mask=mask,        
    annot=True,     
    fmt=".1f",       
    center=0,        
    cmap="RdBu",    
    square=True,     
    cbar=True,        
    linewidths=.5,   
    ax=ax
    )

    ax.set_title("Mapa de calor - Correlação (triângulo inferior)")
    plt.tight_layout()

    # 16
    fig.savefig('heatmap.png')
    return fig
