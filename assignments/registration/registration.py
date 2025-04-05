import numpy as np
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from sklearn.linear_model import RANSACRegressor
from sklearn.ensemble import IsolationForest

def paired_point_matching(source, target):
    """
    Calculates the transformation T that maps the source to the target point clouds
    :param source: A N x 3 matrix with N 3D points
    :param target: A N x 3 matrix with N 3D points
    :return:
        T: 4x4 transformation matrix mapping source to target
        R: 3x3 rotation matrix part of T
        t: 1x3 translation vector part of T
    """
    assert source.shape == target.shape
    T = np.eye(4)
    R = np.eye(3)
    t = np.zeros((1, 3))

    ## TODO: your code goes here
    
    
    # ransac = RANSACRegressor()
    # ransac.fit(source, target)
    # inlier_mask = ransac.inlier_mask_
    # source = source[inlier_mask]
    # target = target[inlier_mask]

    # 1. For each point source --> MATCH: R* source + t = destination
    # 1.1. PPM: Construct covariance matrix M = Q*P^T
    μ_l = np.mean(source,axis=0)
    μ_r = np.mean(target,axis=0)

    Q = target -μ_r
    P = source - μ_l 
    M = np.dot(P.T,Q)

    # 1.2 PPM: apply SVD s.t. M = U*W*V^T
    U,_,Vt = np.linalg.svd(M,full_matrices=False)

    # 1.3 PPM: R = V*U^T 
    R = np.dot(U,Vt)

    # 1.4 PPM: t = μ_r - R*μ_L
    t = μ_r - np.dot(R,μ_l)

    T[:3,:3] = R 
    T[0:3,3] = t     

    return T, R, t


def find_nearest_neighbor(src, dst):
    """
    Finds the nearest neighbor of every point in src in dst
    :param src: A N x 3 point cloud
    :param dst: A N x 3 point cloud
    :return: the index and distance of the closest point
    """

    ## TODO: replace this by your code
    tree = KDTree(dst,leaf_size=40)

    # function that returns distance + index of the nearest neighbor
    return tree.query(src,k=1)


def icp(source, target, init_pose=None, max_iterations=10, tolerance=0.0001):
    """
    Iteratively finds the best transformation that mapps the source points onto the target
    :param source: A N x 3 point cloud
    :param target: A N x 3 point cloud
    :param init_pose: A 4 x 4 transformation matrix for the initial pose
    :param max_iterations: maximum number of iterations to perform, default is 10
    :param tolerance: maximum allowed error, default is 0.0001
    :return: A 4 x 4 rigid transformation matrix mapping source to target,
            the distances between each paired points, and the registration error
    """
    T = np.eye(4)
    distances = 0
    error = np.finfo(float).max
    print("[ERROR]:",error)
    print("[SOURCE]:",source.shape)
    print("[TARGET]:",target.shape)


    ## TODO: Your code goes here
    T = init_pose

    for iter in range(max_iterations):

        # 1. for each point in source cloud, match closest in reference point cloud
        dist, index = find_nearest_neighbor(source,target)

        # 2.1 estimate the combination of R and t using RMSE
        ss_target = target[index].reshape((source.shape))

        # 2.2 This step may also involve weighting points and rejecting outliers prior to alignment.
        outliers = IsolationForest(contamination=0.1, random_state=42).fit(target).predict(source)
        filtered_index = [index for index,outlier in enumerate(outliers) if outlier == 1]
        print(source[filtered_index].shape)
        print(source.shape)
        T,R, t = paired_point_matching(source[filtered_index],ss_target)


        # 2. Estimate the combination of rotation and translation using a 
        # RMSE point to point distance metric minimization technique which 
        # will best align each source point to its match found in the previous step. 
        # This step may also involve weighting points and rejecting outliers prior to 
        # alignment.
        
        error = np.sum(dist) / source.shape[0]

        print(f"[Iter {iter}]: RMSE={error}")
        if error < tolerance:
            break
        
        # 4. Transform source points using obtained transformation
        source = np.dot(source,R) + t
        

    return T, distances, error


def get_initial_pose(source, target):
    """
    Calculates an initial rough registration
    (Optionally you can also return a hand picked initial pose)
    :param source: A N x 3 point cloud
    :param target: A N x 3 point cloud
    :return: An initial 4 x 4 rigid transformation matrix mapping source to target
    """
    T = np.eye(4)

    ## TODO: Your code goes here
    
    # 1. PCA 
    source_centered = source - np.mean(source, axis=0)
    target_centered = target - np.mean(target, axis=0)
    
    covariance_source = np.cov(source_centered, rowvar=False)
    covariance_target = np.cov(target_centered, rowvar=False)
    
    source_pc = PCA().fit(covariance_source).components_
    target_pc = PCA().fit(covariance_target).components_
    
    R = np.dot(target_pc.T, source_pc)
    
    T[:3,:3] = R 

    # 2. centroid alignment
    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)
    translation = centroid_target - centroid_source
    
    T[:3, 3] = translation


    return T

