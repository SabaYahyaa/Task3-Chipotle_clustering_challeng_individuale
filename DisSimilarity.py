import numpy as np



def dissimilarities(all_observation, cluster_centers, distance_type='Eculidean'):
    """
    Finds the dissimilaries between each observation and cendroids.
    The dissimilarities is the distance between the observation (features that have specific cluster_label),
    and the sumation of all centroids except the one that has the same cluster_labe.
    The dissimilarities is a measure of how far the current cluster_type to all other clusters.
    :param all_observation: is a numpy array that contains all used features in the fitting with extra colums which is the cluster_laber from the clustering model
    :param cluster_conters: is the attribute from the clustering model (the cendrotic)
    :param distance_type: is the type of distance that you want, Eculidean, L1 norm, squared distance, inverse squared distance
    :return: the dissimilarities vector
    """
    dissimilarities=[]
    for observation in all_observation:
        dist=0
        # print(observation)
        # print(observation[-1])
        # import pdb
        # pdb.set_trace()
        for c in range(0, len(cluster_centers)):
            if c==observation[-1]:
                continue
            if distance_type=='Eculidean':  #L2-norm
                dist +=np.linalg.norm(cluster_centers[c] - observation[0:-1])
            elif distance_type=="Mahattan": #L1-norm
                dist += np.linalg.norm(cluster_centers[c] - observation[0:-1], ord=1)
            elif distance_type=='squared':
                dist += (np.linalg.norm(cluster_centers[c] - observation[0:-1]))**2
            elif distance_type=='inv_squared':
                dist += 1/float((np.linalg.norm(cluster_centers[c] - observation[0:-1]))**2)

        dissimilarities.append(dist)
    return (dissimilarities)

if __name__=="__main__":
    print("hhhh")
