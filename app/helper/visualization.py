import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import seaborn as sns
import numpy as np
from sklearn import metrics
import shap

def beeswarm_plot(df,df_shap,list_X_num,list_X_cat) :
    df[list_X_num] = df[list_X_num].astype(float)


    df_col = df[list_X_num].copy()

    scaler = MinMaxScaler(feature_range=(-1,1))
    df_col = scaler.fit_transform(df_col)
    df_col = pd.DataFrame(df_col,columns=list_X_num )
    df_col[list_X_cat] = 0
    df_col.reset_index(inplace=True)
    df_col = df_col.melt(value_name='Feature Value',id_vars='index')

    df_shap = df_shap.melt(value_name='SHAP Value',id_vars='index')
    df_plot = df_shap.merge(df_col,on=['index','variable'],how='inner')

    fig, ax = plt.subplots(1,2,figsize=(6, 5),gridspec_kw={'width_ratios': [20, 1]})
    sns.stripplot(df_plot,
                x='SHAP Value',
                y='variable',
                hue='Feature Value',
                alpha = 0.8,
                size=5,
                palette= 'coolwarm',
                legend = False,
                jitter=0.3,ax=ax[0])
    ax[0].axvline(0)
    ax[0].spines[['right', 'top']].set_visible(False)
    ax[0].spines['left'].set_bounds((0, 9))
    ax[0].set_xlabel('SHAP Value (impact on model output)')
    ax[0].set_ylabel(None)
    ####


    # Create a sample colormap and normalization
    cmap = cm.coolwarm
    norm = Normalize(vmin=-1, vmax=1)

    # Create the color bar using the ScalarMappable
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])

    # Add the color bar to the right side of the figure
    cbar = plt.colorbar(sm, orientation='vertical', cax=ax[1])

    cbar.set_ticks([-0.95, 0.95])
    cbar.set_ticklabels(['Low', 'High'])
    cbar.outline.set_edgecolor('none')
    ax[1].tick_params(axis=u'both', which=u'both',length=0)
    cbar.set_label('Feature Value')
    return fig,ax

def prediction_distribution_plot():
    fig = plt.figure()
    sample_neg = np.random.normal(0.3,0.1,1500)
    sample_pos = np.random.normal(0.7,0.2,1000)

    sample_neg = np.clip(sample_neg,0,1)
    sample_pos = np.clip(sample_pos,0,1)

    plt.hist(sample_neg, density=True,bins=50,histtype='step',label='Prediction distribution of False')
    plt.hist(sample_pos, density=True,bins=50,histtype='step',label='Prediction distribution of True')
    plt.legend()
    plt.title("Prediction distribution (based on 10 fold Cross Validation)")

    return fig

def confusion_matrix_plot():
    actual = np.random.binomial(1,.5,size = 1000)
    predicted = np.random.binomial(1,.5,size = 1000)

    confusion_matrix = metrics.confusion_matrix(actual, predicted)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

    fig = cm_display.plot(cmap='mako').figure_
    plt.title("Confusion Matrix")
    return fig

def threshold_performance_plot():
    actual = np.random.binomial(1,.5,size = 1000)
    predicted = np.random.uniform(0,1,size = 1000)

    list_threshold = np.arange(0,1,0.05)
    list_metrics = []
    for threshold in list_threshold:
        prec,rec,f1,_= metrics.precision_recall_fscore_support(actual,predicted>threshold,average='binary',pos_label=1)
        list_metrics.append([prec,rec,f1])
        
    fig = plt.figure()
    plt.plot(list_threshold,list_metrics,label=['Precision','Recall','F1 Score'])
    plt.legend()
    plt.title("Model performances across different thresholds")
    plt.xlabel("Decision Boundary / Prediction Threshold")
    plt.ylabel("Value")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    return fig

def roc_auc_plot():
    actual = np.random.binomial(1,.5,size = 1000)
    predicted = np.random.uniform(0,1,size = 1000)
    fpr, tpr, threshold = metrics.roc_curve(actual, predicted)
    roc_auc = metrics.auc(fpr, tpr)

    fig = plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    return fig

def shap_pdp_plot(shap_values,col,list_X_cat):
    fig, ax = plt.subplots()
    if col not in list_X_cat : 
        shap.plots.scatter(shap_values[:,col],ax=ax)
    else : 
        val_shap = shap_values[:,col].values
        val_data = shap_values[:,col].data
        per_group_average = pd.DataFrame({"val":val_shap,'cat':val_data}).groupby("cat")['val'].mean().sort_values(ascending=False).to_dict()
        category,value = zip(*list(per_group_average.items()))
        sns.stripplot(y=val_data,
                        x=val_shap,
                        alpha = 0.9,
                        size=5,
                        legend = False,
                        hue=val_data,
                        jitter=0.3,zorder=0,order=category,ax=ax)
        sns.scatterplot(x=value,y=category,marker='|',s=500,c='k',label='Average SHAP',ax=ax)
    return fig

def total_error_plot(test_df):
    fig = sns.displot(test_df['error'],kde=True)
    return fig

def error_per_fold_plot(test_df):
    fig, ax = plt.subplots(figsize=(4,4.3))
    ax = sns.boxplot(data=test_df,x='fold',y='error')
    ax.axhline(0,c='red')
    ax.set_ylim(-2,2)
    ax.set_xlabel("Cross Validation Fold")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals in 10 fold cross validation")
    return fig

def residual_per_feature_plot(test_df,col,list_X_cat):
    fig, ax = plt.subplots()
    if col not in list_X_cat : 
        ax = sns.scatterplot(test_df,x=col,y='error')
        ax.set_ylim(-2,2)
        ax.axhline(0,c='red')
    else :
        ax = sns.stripplot(test_df,y=col,x='error')
        ax.set_xlim(-2,2)
        ax.axvline(0,c='red')
    return fig