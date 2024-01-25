import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df_goups = pd.read_csv('../data/PeopleAtPark.csv')

def GetGroups(df):
    df_groups = df.drop(['id'], axis = 1)
    kmeans = KMeans(n_clusters=12, random_state=0).fit(df_groups)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    df['cluster'] = labels
    
    groups = pd.DataFrame(columns=['id','timeSpentInPark','daysAtPark','numberCheckin',
                               'distanceTraveled','numberThrill','numberKiddie',
                               'numberEveryone','numberShows','numberInfo','sizeOfGroup', 'HowCommon'])
    group = 0
    for centroid in centroids:
        groups = groups.append( {"id": group,
                    "timeSpentInPark":round(centroid[0]),
                    "daysAtPark": round(centroid[1],2),
                    "numberCheckin": round(centroid[2],1),
                    "distanceTraveled": round(centroid[3],2),
                    "numberThrill": round(100*centroid[4]/centroid[2],4),
                    "numberKiddie": round(100*centroid[5]/centroid[2],4),
                    "numberEveryone":round(100*centroid[6]/centroid[2],4),
                    "numberShows":round(100*centroid[7]/centroid[2],4),
                    "numberInfo":round(100*centroid[8]/centroid[2],4),
                    "sizeOfGroup":np.count_nonzero(labels == group),
                    "HowCommon": round(100 *  np.count_nonzero(labels == group) / len(labels),2)
                        },ignore_index=True)
        group += 1
    return groups

groups = GetGroups(df_goups)

groups.head()

