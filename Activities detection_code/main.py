import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from scipy.signal import find_peaks
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import check_random_state
from scipy import signal
import seaborn as sns
import scipy as sp
import pickle

#matplotlib.use('tkagg')

cm = plt.get_cmap('gist_rainbow')
line_styles=['solid','dashed','dotted']

def generateDF(filedir,colnames,sensors,patients,activities,slices):

    x=pd.DataFrame()
    for pat in patients:
        for a in activities:
            subdir='a'+f"{a:02d}"+'/p'+str(pat)+'/'
            for s in slices:
                filename=filedir+subdir+'s'+f"{s:02d}"+'.txt'
                #print(filename)
                x1=pd.read_csv(filename,usecols=sensors,names=colnames)
                x1['activity']=a*np.ones((x1.shape[0],),dtype=int)
                x=pd.concat([x,x1], axis=0, join='outer', ignore_index=True,
                            keys=None, levels=None, names=None, verify_integrity=False,
                            sort=False, copy=True)
    return x

def main(filedir, n_activities, Nslices, patients, n_sens, actNamesShort, sensNames):

    window_size = 5*25
    step_size = 125
    NAc = 19

    activities = list(range(1, NAc+1))  # list of indexes of activities to plot
    Num_activities = len(activities)
    actNamesSub = [actNamesShort[i - 1] for i in activities]  # short names of the selected activities
    sensors = list(range(n_sens))  # list of sensors
    sensNamesSub = [sensNames[i] for i in sensors]  # names of selected sensors

    # Ntot=60 #total number of slices

    slices = list(range(1, Nslices + 1))  # first Nslices to plot
    fs = 25  # Hz, sampling frequency
    samplesPerSlice = fs * 5  # samples in each slice

    x = generateDF(filedir,sensNamesSub,sensors,patients,activities,slices)

    # #################################### Generare df ###########################################

    ##todo: SCOMMENTARE

    final = pd.DataFrame()
    finestre_diz = {}
    for a in x['activity'].unique():
        x_activity = x[x['activity']==a]

        #todo: fare in parallelo
        finestre = []
        print(a) #Per vedere avanzamento
        for i in range(0,  x_activity.shape[0], step_size):
            tmp_diz = {}
            for sens in x_activity.columns[:-1]:
                p = pd.DataFrame(x_activity[sens].values[i: i + window_size])

                tmp_diz[f"media_{sens}"] = p.apply(lambda x: x.mean())
                tmp_diz[f"std_{sens}"] = p.apply(lambda x: x.std())
                tmp_diz[f"aad_{sens}"] = p.apply(lambda x: np.mean(np.absolute(x - np.mean(x))))
                tmp_diz[f"min_{sens}"] = p.apply(lambda x: x.min())
                tmp_diz[f"max_{sens}"] = p.apply(lambda x: x.max())
                tmp_diz[f"maxmin_diff_{sens}"] = tmp_diz[f"max_{sens}"] - tmp_diz[f"min_{sens}"]
                tmp_diz[f"median_{sens}"] = p.apply(lambda x: np.median(x))
                tmp_diz[f"mad_{sens}"] = p.apply(lambda x: np.median(np.absolute(x - np.median(x))))
                tmp_diz[f"IQR_{sens}"] = p.apply(lambda x: np.percentile(x, 75) - np.percentile(x, 25))
                tmp_diz[f"neg_count_{sens}"] = p.apply(lambda x: np.sum(x < 0))
                tmp_diz[f"pos_count_{sens}"] = p.apply(lambda x: np.sum(x > 0))
                tmp_diz[f"above_mean_{sens}"] = p.apply(lambda x: np.sum(x > x.mean()))
                tmp_diz[f"peak_count_{sens}"] = p.apply(lambda x: len(find_peaks(x)[0]))
                tmp_diz[f"skewness_{sens}"] = p.apply(lambda x: stats.skew(x))
                tmp_diz[f"kurtosis_{sens}"] = p.apply(lambda x: stats.kurtosis(x))
                tmp_diz[f"energy_{sens}"] = p.apply(lambda x: np.sum(x**2/window_size))


            tmp_diz['classe'] = a
            tmp_window = pd.DataFrame(data=tmp_diz)
            final = pd.concat([final, tmp_window])

    # final.reset_index(drop=True, inplace=True)
    # final.to_pickle("./features.pkl")

    # #################################### Usare df ########################################### #

    # with open("features.pkl", "rb") as f:
    #     final = pickle.load(f)

    subset = final[[c for c in final.columns if "media_" in c]+["classe"]]

    # X = subset.iloc[:, :-1]
    # y = subset.iloc[:, -1]

    X = final.iloc[:, :-1]
    y = final.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42, stratify=y)
    # ################################################## Tree class
    n_feat = 20

    model = ExtraTreesClassifier(random_state=np.random.RandomState(seed))
    #model = RandomForestClassifier(random_state=np.random.RandomState(seed))
    model.fit(X_train, y_train)
    #print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(n_feat).plot(kind='barh')
    plt.show()

    top_imp_features = feat_importances.nlargest(n_feat).index
    print(top_imp_features)
    subset = X_train[[c for c in final.columns if c in top_imp_features]]
    X_train = subset.iloc[:, :]

    subset = X_test[[c for c in final.columns if c in top_imp_features]]
    X_test = subset.iloc[:, :]

    # ################################################ Corr method
    # n_feat = 30
    #
    # # get correlations of each features in dataset
    # X_train["classe"] = y_train
    # corrmat = X_train.corr()
    # try:
    #     X_train.drop(columns=["classe"], inplace=True)
    # except:
    #     pass
    # abs_corr_with_target = corrmat["classe"].apply(lambda x: np.abs(x))
    # top_corr = abs_corr_with_target.drop("classe").nlargest(n_feat)
    # top_corr_features = abs_corr_with_target.drop("classe").nlargest(n_feat).index
    #
    # subset = X_train[[c for c in final.columns if c in top_corr_features]]
    # X_train = subset.iloc[:, :]
    #
    # subset = X_test[[c for c in final.columns if c in top_corr_features]]
    # X_test = subset.iloc[:, :]
    # ################################################

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)

    kmeans = KMeans(n_clusters=19, random_state=0, n_init="auto").fit(X_train_sc)
    y_kmeans = kmeans.predict(X_test_sc)
    y_kmeans_tr = kmeans.predict(X_train_sc)

    centroidi = {}
    for classe in y_train.unique():
        centroidi[f"cluster_{classe}"] = X_train[y_kmeans_tr == classe-1].mean()
        centroidi[f"classe_{classe}"] = X_train[y_train == classe].mean()

    mapping = {}  # cluster: classe
    for cluster in y_train.unique():
        distances = []
        for classe in y_train.unique():
            distances.append(np.linalg.norm(centroidi[f"cluster_{cluster}"].values - centroidi[f"classe_{classe}"].values))

        mapping[cluster-1] = y_train.unique()[np.array(distances).argsort()[0]]

    y_train2 = y_train.reset_index(drop=True)
    # y_train[y_train == 19]
    # y_kmeans_tr[y_train == 19]

    # mapping sul test

    centroidi_te = {}
    for classe in y_test.unique():
        centroidi_te[f"cluster_{classe}"] = X_test[y_kmeans == classe - 1].mean()
        #centroidi_te[f"classe_{classe}"] = X_test[y_test == classe].mean()

    mapping_te = {}  # cluster: classe
    for cluster in y_test.unique():
        distances_te = []
        for classe in y_test.unique():
            distances_te.append(np.linalg.norm(centroidi_te[f"cluster_{cluster}"].values - centroidi[f"classe_{classe}"].values))  ## centroidi_te

        mapping_te[cluster - 1] = y_test.unique()[np.array(distances_te).argsort()[0]]

    y_test2 = y_test.reset_index(drop=True)
    # y_test[y_test == 13]
    # y_kmeans[y_test == 13]

    y_kmeans_tr_mapped = pd.Series([mapping[y_pred] for y_pred in y_kmeans_tr])
    y_kmeans_mapped = pd.Series([mapping_te[y_pred] for y_pred in y_kmeans])

    # accuracy sul train
    print("Accuracy on train:", accuracy_score(y_train2, y_kmeans_tr_mapped))

    # accuracy sul test
    print("Accuracy on test:", accuracy_score(y_test2, y_kmeans_mapped))

    # Confusion matrix train

    labels = ['sitting', 'standing', 'lying.ba', 'lying.ri', 'asc.sta', 'desc.sta', 'stand.elev', 'mov.elev',
              'walk.park', 'walk.4.fl', 'walk.4.15', 'run.8', 'exer.step', 'exer.train',
              'cycl.hor', 'cycl.ver', 'rowing', 'jumping', 'play.bb']

    conf_matrix_tr = confusion_matrix(y_train2, y_kmeans_tr_mapped, normalize='true')
    sns.heatmap(conf_matrix_tr, xticklabels=labels, yticklabels=labels, annot=True, linewidths=0.1, fmt='.1f', cmap='YlGnBu')
    plt.title("Confusion matrix train", fontsize=15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    acc_tr = []
    for i in range(19):
        acc_tr.append(100 * (conf_matrix_tr[i,i]) / sum(conf_matrix_tr[i,:]))
    print(acc_tr)

    # Confusion matrix test

    # conf_matrix = confusion_matrix(y_test2, y_kmeans_mapped)
    # sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, linewidths=0.1, fmt='d', cmap = 'YlGnBu')
    conf_matrix = confusion_matrix(y_test2, y_kmeans_mapped, normalize='true')
    sns.heatmap(conf_matrix, xticklabels=labels, yticklabels=labels, annot=True, linewidths=0.1, fmt='.1f', cmap = 'YlGnBu')
    plt.title("Confusion matrix test", fontsize = 15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    acc_te = []
    for i in range(19):
        acc_te.append(100 * (conf_matrix[i, i]) / sum(conf_matrix[i, :]))
    print(acc_te)

    # ############################################### CENTROIDI ######################################### #

    stdpoints = np.zeros((NAc, n_feat))  # variance in cluster for each sensor
    centroids = np.zeros((NAc, n_feat))  # centroids for all the activities

    # x_full = pd.DataFrame(X_train_sc, columns=X_train.columns)
    # x_full = pd.concat([x_full, pd.DataFrame(X_test_sc, columns=X_train.columns)])
    # y_mapped_full = pd.concat([y_kmeans_tr_mapped, y_kmeans_mapped])
    # x_full["classe"] = y_mapped_full

    ######################
    X = final.iloc[:,:]
    classi = X["classe"]
    subset = X[[c for c in final.columns if c in top_imp_features]]
    X = subset.iloc[:,:]
    X = X.assign(classe=classi)

    for i in range(1, NAc + 1):
        x_tmp = X[X["classe"]==i]
        x_tmp = x_tmp.drop(columns=['classe'])
        centroids[i - 1, :] = x_tmp.mean().values
        stdpoints[i - 1] = np.sqrt(x_tmp.var().values)

    d = np.zeros((NAc, NAc))
    for i in range(NAc):
        for j in range(NAc):
            d[i, j] = np.linalg.norm(centroids[i] - centroids[j])  # sostituisco con centroidi di xfull e calcolo come prof


    plt.matshow(d)
    plt.colorbar()
    plt.xticks(np.arange(NAc), actNamesShort, rotation=90)
    plt.yticks(np.arange(NAc), actNamesShort)

    # plt.title('Between-centroids distance')

    # %% compare minimum distance between two centroids and mean distance from a cluster point and its centroid

    dd = d + np.eye(NAc) * 1e6  # remove zeros on the diagonal (distance of centroid from itself)
    dmin = dd.min(axis=0)  # find the minimum distance for each centroid
    dpoints = np.sqrt(np.sum(stdpoints ** 2, axis=1))
    plt.figure()
    plt.plot(dmin, label='minimum centroid distance')
    plt.plot(dpoints, label='mean distance from points to centroid')
    plt.grid()
    plt.xticks(np.arange(NAc), actNamesShort, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ################################# centroids prima della feature engineering ##############

    centroids = np.zeros((NAc, len(sensors)))  # centroids for all the activities
    stdpoints = np.zeros((NAc, len(sensors)))  # variance in cluster for each sensor
    for i in range(1, NAc + 1):
        activities = [i]
        x = generateDF(filedir, sensNamesSub, sensors, patients, activities, slices)
        x = x.drop(columns=['activity'])
        centroids[i - 1, :] = x.mean().values
        stdpoints[i - 1] = np.sqrt(x.var().values)

    d = np.zeros((NAc, NAc))
    for i in range(NAc):
        for j in range(NAc):
            d[i, j] = np.linalg.norm(centroids[i] - centroids[j])

    plt.matshow(d)
    plt.colorbar()
    plt.xticks(np.arange(NAc), actNamesShort, rotation=90)
    plt.yticks(np.arange(NAc), actNamesShort)

    # plt.title('Between-centroids distance')

    # %% compare minimum distance between two centroids and mean distance from a cluster point and its centroid

    dd = d + np.eye(NAc) * 1e6  # remove zeros on the diagonal (distance of centroid from itself)
    dmin = dd.min(axis=0)  # find the minimum distance for each centroid
    dpoints = np.sqrt(np.sum(stdpoints ** 2, axis=1))
    plt.figure()
    plt.plot(dmin, label='minimum centroid distance')
    plt.plot(dpoints, label='mean distance from points to centroid')
    plt.grid()
    plt.xticks(np.arange(NAc), actNamesShort, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print(1)
    plt.close('all')
    filedir='C:\\Users\\feder\\PycharmProjects\\Lab04_ICT4HealthV2\\project\\data\\'
    sensNames=[
            'T_xacc', 'T_yacc', 'T_zacc',
            'T_xgyro','T_ygyro','T_zgyro',
            'T_xmag', 'T_ymag', 'T_zmag',
            'RA_xacc', 'RA_yacc', 'RA_zacc',
            'RA_xgyro','RA_ygyro','RA_zgyro',
            'RA_xmag', 'RA_ymag', 'RA_zmag',
            'LA_xacc', 'LA_yacc', 'LA_zacc',
            'LA_xgyro','LA_ygyro','LA_zgyro',
            'LA_xmag', 'LA_ymag', 'LA_zmag',
            'RL_xacc', 'RL_yacc', 'RL_zacc',
            'RL_xgyro','RL_ygyro','RL_zgyro',
            'RL_xmag', 'RL_ymag', 'RL_zmag',
            'LL_xacc', 'LL_yacc', 'LL_zacc',
            'LL_xgyro','LL_ygyro','LL_zgyro',
            'LL_xmag', 'LL_ymag', 'LL_zmag']
    actNames=[
        'sitting',  # 1
        'standing', # 2
        'lying on back',# 3
        'lying on right side', # 4
        'ascending stairs' , # 5
        'descending stairs', # 6
        'standing in an elevator still', # 7
        'moving around in an elevator', # 8
        'walking in a parking lot', # 9
        'walking on a treadmill with a speed of 4 km/h in flat', # 10
        'walking on a treadmill with a speed of 4 km/h in 15 deg inclined position', # 11
        'running on a treadmill with a speed of 8 km/h', # 12
        'exercising on a stepper', # 13
        'exercising on a cross trainer', # 14
        'cycling on an exercise bike in horizontal positions', # 15
        'cycling on an exercise bike in vertical positions', # 16
        'rowing', # 17
        'jumping', # 18
        'playing basketball' # 19
        ]
    actNamesShort=[
        'sitting',  # 1
        'standing', # 2
        'lying.ba', # 3
        'lying.ri', # 4
        'asc.sta' , # 5
        'desc.sta', # 6
        'stand.elev', # 7
        'mov.elev', # 8
        'walk.park', # 9
        'walk.4.fl', # 10
        'walk.4.15', # 11
        'run.8', # 12
        'exer.step', # 13
        'exer.train', # 14
        'cycl.hor', # 15
        'cycl.ver', # 16
        'rowing', # 17
        'jumping', # 18
        'play.bb' # 19
        ]
    ID=247586
    s=ID%8+1
    patients=[s]
    seed = 247586
    cutoff_lp = 2
    cutoff_hp = 0.5
    order = 3

    n_activities = 19
    n_slices = 60
    n_sens = 45

    main(filedir, n_activities, n_slices, patients, n_sens, actNamesShort, sensNames)
