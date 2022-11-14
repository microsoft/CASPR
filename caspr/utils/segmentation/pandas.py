import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm


def check_sparsity(data):
    """Check sparsity in data."""
    for c in data.columns:
        try:
            temp = pd.qcut(data[c], q=10, labels=False, duplicates='drop').value_counts()/data.shape[0]

            # top quantile%, unique value%
            print(c, np.round(temp.values[0], 2), np.round(len(data[c].unique())/data.shape[0], 2))
        except Exception:
            print(c, np.nan, np.round(len(data[c].unique()), 2))


def quantile(df, q=5, col_features=None):
    """Score customers from 0 to 5 based on Engagement metrics.

    Input:
    Output:
    """

    # create quantile scores [0, q] with q interval
    for c in col_features:
        if 'R_' in c:
            df[c+'_q'] = pd.qcut(df[c], q=q+1, labels=range(q, -1, -1), duplicates='drop')
        else:
            df[c+'_q'] = pd.qcut(df[c], q=q+1, labels=range(0, q+1), duplicates='drop')

    df['AvgScore'] = df[[c + '_q' for c in col_features]].mean(axis=1)
    df['AvgScore'].hist(bins=q)
    plt.title('AvgScore')
    plt.show()

    # generate segments
    df['Segment'] = np.nan
    for i in range(1, q+1):
        df.loc[(df.AvgScore <= i) & (df.AvgScore > (i-1)), 'Segment'] = i

    df['Segment'].hist(bins=q)
    plt.title('Segment')
    plt.show()

    return df


def clustering(df, col_features=None, cluster_range=range(2, 10), scaling_option="minmax",
               pca=True, pca_param={'threshold': 0.8, 'show_plot': False},
               default_cluster_size=None, default_cluster_threshold=0.1,
               tsne_plt=True, tsne_sample=1000, removed_outlier=False):
    """Perform Clustering.

    Options to do transformation and PCA before performing clustering
        - featurization
        - find # of clusters
        - fit final model
    """
    inertias = []
    sil_scores = []

    # featurization
    df = apply_scaling(df, col_features, scaling_option, removed_outlier)

    if pca:
        df_features, n_pca, pca = apply_pca(df, col_features=col_features, pca_param=pca_param)  # noqa W0612
    else:
        df_features = df[col_features].values

    # find # of clusters
    for k in tqdm(cluster_range):
        # kc = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=1)
        kc = KMeans(n_clusters=k, random_state=1, n_jobs=4)

        kc.fit(df_features)
        sil_scores.append(silhouette_score(df_features, kc.labels_))
        inertias.append(kc.inertia_)

    results = pd.DataFrame(np.array([cluster_range, inertias, sil_scores]).T)
    results.columns = ['cluster_size', 'inertias', 'sil_scores']
    n_final = cluster_range[np.where(sil_scores == np.max(sil_scores))[0][0]]
    print('optimal cluster size:', n_final, np.round(np.max(sil_scores), 2))

    # fit final model
    if default_cluster_size is not None:
        # kc = KMeans(n_clusters=default_cluster_size, random_state=1)
        sil_score_default = results.loc[results.index == default_cluster_size, 'sil_scores'].values[0]

        if (np.max(sil_scores)/sil_score_default - 1) <= default_cluster_threshold:
            print('default is a good cluster size', sil_score_default, np.max(sil_scores))
            kc = KMeans(n_clusters=default_cluster_size, random_state=1)
        else:
            print('optimal is a better cluster size', sil_score_default, np.max(sil_scores))
            kc = KMeans(n_clusters=n_final, random_state=1)
    else:
        kc = KMeans(n_clusters=n_final, random_state=1)

    kc.fit(df_features)

    df['label'] = kc.labels_

    # score visualization
    if len(cluster_range) > 1:
        _, axes = plt.subplots(1, 2, figsize=(10, 5))
        results.plot(ax=axes[0], x='cluster_size', y='inertias')
        results.plot.bar(ax=axes[1], x='cluster_size', y='sil_scores')
        plt.show()

    # clustering size distribution
    print(df.label.value_counts().to_frame()/df.shape[0])

    # tsne visualization
    if tsne_plt:
        if (tsne_sample > 0) & (tsne_sample < len(kc.labels_)):
            df_tsne = pd.DataFrame(df_features)
            df_tsne['label'] = kc.labels_

            plt_tsne(x=df_tsne.drop(columns=['label']).sample(n=tsne_sample, random_state=1).values,
                     label=df_tsne.sample(n=tsne_sample, random_state=1).label.values)
        else:
            plt_tsne(x=df_features, label=kc.labels_)

    return results, df, kc


def apply_scaling(df, col_features=None, scaling_option=None, removed_outlier=False):
    """Apply Scaling to dataframe."""

    if scaling_option == 'minmax':
        scaler = MinMaxScaler()
        df[col_features] = scaler.fit_transform(df[col_features])
    elif scaling_option == 'qcut':
        for c in col_features:
            df[c] = pd.qcut(df[c], q=100, labels=False, duplicates='drop')
    else:
        pass

    if removed_outlier:
        n_std = 3
        for c in col_features:
            if df[c].dtype != 'object':
                tic_cnt = df.shape[0]
                temp_mean = df[c].mean()
                temp_std = df[c].std()
                df = df[(df[c] <= (temp_mean + n_std*temp_std)) & (df[c] >= (temp_mean - n_std*temp_std))].copy()
                print('remove outlier', c, ':', df.shape[0] - tic_cnt)

    df.reset_index(drop=True, inplace=True)

    return df


def apply_pca(df, col_features=None, pca_param={'threshold': 0.8, 'show_plot': False}):
    """Apply PCA transformation and return # of eigen-vectors based on threshold (20/80 rules)."""

    # normalize the input matrix
    matrix = df[col_features].values
    scaler = StandardScaler()
    scaler.fit(matrix)
    scaled_matrix = scaler.transform(matrix)

    # perform PCA
    pca = PCA()
    pca.fit(scaled_matrix)
    pca_samples = pca.transform(scaled_matrix)

    # # visualize explained variance
    # if pca_param['show_plot']:
    #     fig, ax = plt.subplots(figsize=(10, 5))
    #     sns.set(font_scale=1)
    #     plt.step(range(matrix.shape[1]), pca.explained_variance_ratio_.cumsum(), where='mid',
    #             label='cumulative explained variance')
    #     sns.barplot(np.arange(1,matrix.shape[1]+1), pca.explained_variance_ratio_, alpha=0.5, color = 'g',
    #                 label='individual explained variance')
    #     plt.xlim(0, len(col_features))
    #     ax.set_xticklabels([s if int(s.get_text())%2 == 0 else '' for s in ax.get_xticklabels()])
    #     plt.ylabel('Explained variance', fontsize = 14)
    #     plt.xlabel('Principal components', fontsize = 14)
    #     plt.legend(loc='best', fontsize = 13)

    # define n_pca based on the threshold
    n_pca = np.where(pca.explained_variance_ratio_.cumsum() > pca_param['threshold'])[0][0] + 1
    print('# of pca components:', n_pca, '/', scaled_matrix.shape[1])
    print('# of variance explained:', pca.explained_variance_ratio_.cumsum()[n_pca-1])

    # see loadings of the main components
    df_pca_components = pd.DataFrame(pca.components_, columns=col_features)
    plt_bar(df_pca_components.head(n_pca).copy(), ncols=3, figsize=(10, 10), title='PCA ')

    # extract the main transformed features
    df_pca = pca_samples[:, 0:n_pca]

    return df_pca, n_pca, pca


def profiling(df, label, col_features, col_dropped=[]):
    """Profile Dataframe using heatmap.

    Heat-map around KPIs: absolute & relative
    - Useful technique to identify relative importance of each segment's attribute
    - Calculate average values of each cluster
    - Calculate average values of population
    - Calculate importance score by dividing them and subtracting 1
    (ensures 0 is returned when cluster average equals population average)

    col_dropped: automatic or apply min-max scaling to features [TO-DO]
    """
    df['Segment'] = label

    # classifying cat vs. cont features
    cat_features = []
    cont_features = []
    for x in col_features:
        if df[x].dtypes == 'object':
            cat_features.append(x)
        else:
            cont_features.append(x)

    # customer counts
    df_count = df.groupby('Segment')[cont_features[0]].count()
    df_count.loc['All'] = df_count.sum()
    df_count = df_count.to_frame()
    df_count.columns = ['Customers']
    df_count['Customers%'] = df_count.Customers/df_count.Customers.values[-1]*100

    # numerical features
    df_cont = df.groupby('Segment')[cont_features].mean()
    df_cont.loc['All'] = df_cont.mean()

    # categorical features
    df_cat = pd.DataFrame()
    for c in cat_features:
        df_pivot = df.pivot_table(index='Segment',
                                  columns=c, values=cont_features[0], aggfunc='count')
        df_pivot.loc['All'] = df_pivot.sum()
        df_pivot[df_pivot.columns] = df_pivot.values / df_pivot.sum(axis=1).values.reshape(-1, 1)*100
        df_cat = pd.concat([df_cat, df_pivot], axis=1)

    # combine results and calcuate relative importance
    result_profile = pd.concat([df_count, df_cont, df_cat], axis=1)
    temp_all = result_profile.loc['All']
    result_profile.drop('All', inplace=True)
    result_profile.sort_index(ascending=False, inplace=True)
    result_profile.loc['All'] = temp_all

    relative_imp = result_profile/result_profile.loc['All'] - 1
    relative_imp.drop('All', inplace=True)

    # visualization - heatmap
    temp = relative_imp.drop(columns=['Customers', 'Customers%'] + col_dropped).copy()
    plt_heatmap(temp, x_labels=temp.columns, y_labels=temp.index)

    # visualization - barchart by clusters
    plt_bar(temp)

    return relative_imp, result_profile


def plt_tsne(x, label):
    """Visualize TSNE."""
    tic = time.time()
    x_embedded = TSNE(n_components=2).fit_transform(x)
    print('tsne takes time: ', time.time() - tic)

    vis_x = x_embedded[:, 0]
    vis_y = x_embedded[:, 1]

    fig = plt.figure(figsize=(12, 8)) # noqa W0612
    plt.scatter(vis_x, vis_y, c=label, cmap=plt.cm.get_cmap("jet", 256))
    plt.colorbar(ticks=range(256))
    plt.clim(-0.5, 9.5)
    plt.show()


def plt_heatmap(data, x_labels, y_labels):
    """Plot Heatmap."""
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([1, 1, 1.1, 1.1])

    plt.imshow(data, cmap='Blues', interpolation='nearest')
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=60)
    plt.colorbar()
    plt.show()


def plt_bar(data, ncols=3, figsize=(10, 10), title='Segment'):
    """Plot bars."""

    data.dropna(axis=1, inplace=True)
    nrows = int(np.ceil(data.shape[0]/ncols))
    xlim_min = data.min().min()
    xlim_max = data.max().max()

    fig = plt.figure(figsize=figsize)
    for i in range(data.shape[0]):
        temp = data.iloc[i]
        ax = plt.subplot(nrows, ncols, i+1)
        ax.barh(range(len(temp)), temp.values, align='center')
        if title is not None:
            ax.set_title(title + str(data.index[i]))

        plt.xticks(rotation=45)
        plt.xlim(xlim_min, xlim_max)

        if i % ncols == 0:
            ax.set_yticks(range(len(temp)))
            ax.set_yticklabels(temp.index)

    fig.tight_layout()
    plt.show()


def generate_segmentation_graphs(combined_df, profile_features,
                                 emb_features, use_profile=False, use_embedding=False):
    """Generate segmentation graphs.

    combined_df - the dataframe containing embeddings and profile features
                    with feature names as the column names
    profile_features - the names of all the profile features in the data

    emb_features - name of the embedding features, by default they should be 'dim_0', 'dim_1'...

    use_profile - boolean flag to determine if we use the profile features or not

    use_embedding - boolean flag to determine if we use the embedding values
    """

    # importlib.reload(segmentation_utils)

    df_emb = combined_df
    # Need to remove the dimensions of the embedding which have only onevalue if we use scaling
    col_one = []
    for col in emb_features:
        if df_emb[col].nunique() == 1:
            col_one.append(col)
    df_emb = df_emb.drop(columns=col_one, axis=1)

    emb_featuresN = []  # noqa: C0103
    for item in emb_features:
        if item not in col_one:
            emb_featuresN.append(item)

    emb_features = emb_featuresN

    plt_heatmap(df_emb[emb_features].corr(), emb_features, emb_features)
    df_emb[emb_features].describe()

    features_to_use = []
    if use_profile:
        features_to_use = profile_features
    if use_embedding:
        features_to_use = emb_features

    if use_embedding and use_profile:
        features_to_use = profile_features + emb_features

    n = 5000
    data_c = df_emb.sample(n=n, random_state=1).copy()

    results, df, kc = clustering(df=data_c.copy(),
                                 col_features=features_to_use, cluster_range=range(2, 9), scaling_option='qcut',
                                 pca=True, pca_param={'threshold': 0.8, 'show_plot': False},
                                 default_cluster_size=None, default_cluster_threshold=0.1,
                                 tsne_plt=True, tsne_sample=1000, removed_outlier=False)

    col_features = emb_features + profile_features
    relative_imp, result_profile = profiling(data_c.copy(), kc.labels_, col_features, col_dropped=[])


def generate_combined_df(embedding_data=None, profile_data: pd.DataFrame = None):
    """Generate Combined DF.

    embedding data - The numpy array containing the embeddings in the NxM format where
                    N = number of data entries, M = embedding dimension

    profile data - The dataframe containing all the profile features, with column names
                    equal to the featur names

    """
    if embedding_data is None:
        profile_data.reset_index(drop=True, inplace=True)
        return profile_data

    emb_dim = embedding_data.shape[1]
    column_list = []
    for i in range(emb_dim):
        column_list.append('dim_' + str(i))
    emb_df = pd.DataFrame(embedding_data, columns=column_list)
    emb_df.reset_index(drop=True, inplace=True)

    if profile_data is None:
        return emb_df

    profile_data.reset_index(drop=True, inplace=True)
    final_df = pd.concat([profile_data, emb_df], axis=1)
    return final_df
