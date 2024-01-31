import numpy as np
import open3d as o3d
from matplotlib.image import imread
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ellipse import LsqEllipse
from matplotlib.patches import Ellipse
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
import matplotlib as mpl
from shapely.geometry.point import Point
import shapely.affinity
import shapely
from copy import deepcopy
import os
from polynomial_surface import polyfit2d, polyval2d, RMSE
from pathos.multiprocessing import ProcessingPool as PathosPool
from alphashape import alphashape

# class to hold tide data
#-------------------------------------------------------------
class PointCloud:
    def __init__(self, ply_path = None, points = None, colors = None, name = None, labels = None):
        """

        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Class object to describe dense cloud
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        :param ply_path: path to .ply file (string, Required)
        :param name: name of the point cloud (string, Optional)


        """
        if ply_path != None:
            pcd = o3d.io.read_point_cloud(ply_path)
            self.labels = np.zeros_like(np.arange(len(np.asarray(pcd.points))))

        else:
            pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points))
            if type(colors) == np.ndarray:
                pcd.colors = o3d.utility.Vector3dVector(colors)
            if type(labels) != np.ndarray:
                if labels == None:
                    self.labels = np.zeros_like(np.arange(len(points)))
            else:
                self.labels = labels

        self.pcd = pcd
        self.arr = np.asarray(pcd.points);
        self.points = self.arr.copy()
        self.points[np.isnan(self.points)] = 0
        self.X = self.points[:,0]
        self.Y = self.points[:,1]
        self.Z = self.points[:,2]

        self.n = len(self.points)
        self.colors = np.asarray(pcd.colors)
        #self.colors[np.isnan(self.points)] = 0

        self.color_dists = {}
        self.point_dists = {}

        self.original_points = self.points.copy()

        self.cluster_params = {'esp':0.05, 'min_samples': 100}

    def calculateColorDist(self, color, color_name = 'a_color'):
        """
        Method to calcualte euclidean distance in color values for each point
        :param color: 3-sized array with decimal (0-1) values referring to RGB (numpy array, Required)
        :param color_name: color name (string, Optional)
        """

        color_dist = np.sum(np.column_stack([(self.colors[:,i]-color[i])**2 for i in range(3)]), axis = 1)**0.5
        self.color_dists[color_name] = color_dist

    def calculateDistToPoint(self, p, point_name = 'a_point'):
        """
        Method to calculate euclidean distance to given point for each point in point cloud
        :param point: 3-sized array with x,y and z of poi (numpy array, Required)
        :param point_name: color name (string, Optional, defaults to 'a_point')

        """

        point_dist = np.sum(np.column_stack([(self.points[:,i]-p[i])**2 for i in range(3)]), axis = 1)**0.5
        self.point_dists[point_name] = point_dist

    def maskOnColorDist(self, color_name = 'a_color', dist = 0.1):
        """
        Method to mask a dense cloud on distance to color
        :param color_name: color name (string, Optional, defaults to 'a_color')
        :param dist: euclidean distance threshold (float, Optional, defaults to 0.1)
        :return: copy of the point cloud object with only the maksed points
        """

        color_dist = self.color_dists[color_name]
        mask = (color_dist < dist)
        pcd_points_masked = self.points[mask, :]
        pcd_colors_masked = self.colors[mask, :]
        labels = self.labels[mask]

        return PointCloud(points = pcd_points_masked, colors = pcd_colors_masked, labels = labels)

    def plot(self, colors = None, ax = None, plot_ax_labels = True,  every_x_point = 1, **kwargs):
        """
        Method to plot
        :param colors: numpy array of n_points length to color the scatter markers (numpy array, Optional, defaults to the color array)
        """

        if type(colors) != np.ndarray:
            if colors == None:
                colors = self.colors

        if ax:
            fig = ax.get_figure()
            gridspecs = ax.get_subplotspec()
            ax.remove()
            ax = fig.add_subplot(gridspecs, projection="3d")
        else:
            fig = plt.figure()
            ax = plt.axes(projection="3d")

        if len(colors.shape) > 1:
            ax.scatter3D(self.points[::every_x_point, 0], self.points[::every_x_point, 1], self.points[::every_x_point, 2], color=colors[::every_x_point],**kwargs)
        else:
            sc = ax.scatter3D(self.points[::every_x_point, 0], self.points[::every_x_point, 1], self.points[::every_x_point, 2], c=colors[::every_x_point], cmap = 'tab20',**kwargs)
            fig.colorbar(sc)

        if plot_ax_labels:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

        return fig, ax

    def transformPCA(self):
        """
        Method to transform the coordinates of the points along its principal components
        """
        pca = PCA(n_components=3)
        pcd_pca = pca.fit_transform(self.points)
        
        return PointCloud(points = pcd_pca, colors = self.colors, labels = self.labels)

    def sample(self, reduction_factor = 100):
        """
        Method to reduce the size of a point cloud.
        :param reduction_factor: only keep 1/reduction_factor of the points

        :return: the reduced point cloud
        
        !!! Warning, a lot of points are lost !!!
        !!! Only use for test purposes !!!
        """

        return PointCloud(points = self.points[::reduction_factor], colors = self.colors[::reduction_factor], labels = self.labels[::reduction_factor])
    
    def cluster(self, technique = 'DBSCAN', param = {'eps':0.015, 'min_samples':10},
                remove_noise = False, return_cluster_clouds = False, only_keep_core_samples = True,
                only_keep_mainclass = False, dimensions = [0,1,2]):
        """
        Method to mask a dense cloud on distance to color which adds an array as attribute.
        That array contains a class label (integer) for each point

        :param param: dictionary with clustering parameters if:
            - technique is DBSCAN:
                * eps : neighbourhood distance of core clusters (float, Optional, defaults to 0.015)
                * dist: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. (float, Optional, defaults to 10)
            - technique is AgglomerativeClustering:
                * distance_threshold : The linkage distance threshold above which, clusters will not be merged (float, Optional)
        :param
        :param remove_noise: boolean to remove all noise after clustering (boolean, Optional, defaults to False)
        """

        X = np.column_stack([self.points[:,i] for i in dimensions])
        if technique == 'DBSCAN':
            db = DBSCAN(eps=param['eps'], min_samples=param['min_samples']).fit(X)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            self.core_samples_mask = core_samples_mask
        elif technique == "AgglomerativeClustering":
            db = AgglomerativeClustering(distance_threshold=param['distance_threshold'], n_clusters = None).fit(X)
        elif technique.lower() == "kmeans":
            db = KMeans(n_clusters=param['n_clusters']).fit(X)
            only_keep_core_samples = False

        self.labels = db.labels_

        self.n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        self.n_noise = list(self.labels).count(-1)

        if remove_noise:
            mask = (self.labels != -1)
            self.points = self.points[mask,:]
            self.colors = self.colors[mask,:]
            self.labels = self.labels[mask]
            self.n = len(self.colors)

        elif only_keep_core_samples:
            self.points = self.points[core_samples_mask, :]
            self.colors = self.colors[core_samples_mask, :]
            self.labels = self.labels[core_samples_mask]
            self.n = len(self.colors)

        if return_cluster_clouds:
            pcls = []
            for label in np.unique(self.labels):
                mask = (self.labels == label)
                pcls.append(self.mask(mask))
            return pcls

        if only_keep_mainclass:
            if self.n_clusters > 0:
                label_freq, _ = np.histogram(self.labels, bins=self.n_clusters)
                most_common_class = np.sort(np.unique(self.labels))[np.argmax(label_freq)]
                class_mask = (self.labels == most_common_class)
                return self.mask(class_mask)

        else:
            return self

    def rotateAxesAlongAngles(self, yaw = 0, pitch = 0, roll = 0, angle = 'degrees'):
        """
        Method to rotate axes
        :param yaw: rotating angle to rotate counter-clockwise around the z-axis (float, Optional, defaults to 0)
        :param pitch: rotating angle to rotate counter-clockwise around the x-axis (float, Optional, defaults to 0)
        :param roll: rotating angle to rotate counter-clockwise around the y-axis (float, Optional, defaults to 0)
        :param angle: degrees or radians (string, Optional, defaults to degrees)
        """
        if angle == 'degrees':
            yaw = np.deg2rad(yaw)
            pitch = np.deg2rad(pitch)
            roll = np.deg2rad(roll)

        # yaw rotation matrix
        R_yaw =np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw), np.cos(yaw),0],[0,0,1]])

        # pitch rotation matrix
        R_pitch = np.array([[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]])

        # roll rotation matrix
        R_roll = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])

        # rotation matrix
        R = R_yaw @ R_pitch @ R_roll

        return self.rotateAxes(R)

    def rotateAxes_slow(self, R, multiprocessing_flag = False, multiprocessing_cores = 6):
        """
        Method to rotate axes
        :param R: Rotation matrix (3x3 np array, Required)
        """

        self.rotated_points = np.zeros_like(self.points)

        def rotateSinglePoint(x,y,z,rot = R):
            rotated_xyz = rot @ np.array([x, y, z])
            x_, y_, z_ = rotated_xyz
            return np.array([x_,y_,z_])

        for i in range(self.n):
            x, y, z = self.points[i, :]
            self.rotated_points[i,:] = rotateSinglePoint(x,y,z, rot = R)
            print('Progress Rotating Pointcloud: %.1f%%' % (100*(i+1)/self.n), end = '\r')

        return PointCloud(points = self.rotated_points, colors = self.colors, labels = self.labels)

    def rotateAxes(self, R, multiprocessing_flag = False, multiprocessing_cores = 6):
        """
        Method to rotate axes
        :param R: Rotation matrix (3x3 np array, Required)
        """

        print('Rotating... ', end = '')
        self.rotated_points = np.zeros_like(self.points)

        rot_X = R[0,0]*self.X + R[0,1]*self.Y + R[0,2]*self.Z
        rot_Y = R[1,0]*self.X + R[1,1]*self.Y + R[1,2]*self.Z
        rot_Z = R[2,0]*self.X + R[2,1]*self.Y + R[2,2]*self.Z
        self.rotated_points = np.column_stack((rot_X, rot_Y, rot_Z))
        print('Ready!')
        return PointCloud(points = self.rotated_points, colors = self.colors, labels = self.labels)

    def mask(self, mask):
        return PointCloud(points=self.points[mask,:], colors=self.colors[mask,:], labels=self.labels[mask])

    def getCentroid(self):
        return np.mean(self.points, axis = 0)

    def fitEllips(self, dimensions = [1,2], calc_score = False):
        """
        Method to fit ellips to 2 dimensions of the point cloud. We strongly recommend to perform a PCA transformation first.
        :param dimensions: 2-element list with indices of the dimensions to calculate the ellipse on (list, Optional, defaults to [1,2])
        :return ellipse_param: parameters describing an ellipse: center, width, height and phi
        """
        X = np.column_stack([self.points[:,i] for i in dimensions])
        reg = LsqEllipse().fit(X)
        center, width, height, phi = reg.as_parameters()

        if calc_score:
            reg_xy = reg.return_fit(n_points=1000)
            ellr = shapely.geometry.LinearRing(reg_xy)

            dist_to_ellipse = np.array([ellr.distance(Point(X[i,:])) for i in range(len(X))])
            self.dist_to_ellipse = dist_to_ellipse
            dist_avg = np.mean(dist_to_ellipse)
            dist_std = np.std(dist_to_ellipse)
            return [center, width, height, phi],[dist_avg, dist_std], ellr
        else:
            return [center, width, height, phi]

    def filterInterval(self, dimensions = [0,1,2], width = 0.9):

        mask = np.ones(len(self.points), dtype = bool)

        for d in dimensions:
            if type(width)==float:
                lower_boundary = np.quantile(self.points[:,d],(1 - width)/2)
                higher_boundary = np.quantile(self.points[:,d],(width + 1)/2)
            elif type(width)==str:
                if width.endswith('std'):
                    factor = float(width.split('x')[0])
                    lower_boundary = np.mean(self.points[:,d])-factor*np.std(self.points[:,d])
                    higher_boundary = np.mean(self.points[:,d])+factor*np.std(self.points[:,d])

            mask *= (self.points[:,d] > lower_boundary)*(self.points[:,d] < higher_boundary)

        return self.mask(mask)

    def merge(self, pointcloud):
        """
        method to merge two point clouds
        :param pointcloud: other PointCloud object (PointCloud object, Required)
        :returns merged_pointcloud: new merged PointCloud
        """

        merged_points = np.vstack((self.points, pointcloud.points))
        merged_colors = np.vstack((self.colors, pointcloud.colors))
        merged_labels = np.concatenate((self.labels, pointcloud.labels))

        return PointCloud(points=merged_points, colors=merged_colors, labels=merged_labels)

    def copy(self):
        return deepcopy(self)

    def clusterAndFitEllips(self, val):
        self_copy = self.copy()
        min_samples = self.cluster_params['min_samples']
        try:
            self_copy = self_copy.cluster(technique = "DBSCAN",param = {'eps':val, 'min_samples': min_samples}, only_keep_core_samples = True, only_keep_mainclass = True, dimensions=[1,2])
            #self_copy = self_copy.cluster(technique = "AgglomerativeClustering",param = {'distance_threshold': val}, only_keep_core_samples = False, only_keep_mainclass = True)
            [center, width, height, phi], [score_mean, score_std], ellr = self_copy.fitEllips(calc_score = True)

        except:
            score_mean = 999

        return score_mean

    def writePLY(self,fn):
        """
        Method to write/save a point cloud to a PLY file
        :param fn: path to file
        """

        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(self.points))
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        o3d.io.write_point_cloud(fn, pcd)

    def normalize(self):
        """
        method to normalize the points to their mean
        :return:
        """

        self.X -= np.mean(self.X)
        self.Y -= np.mean(self.Y)
        self.Z -= np.mean(self.Z)

    def fitSurface(self, order = 3):
        m = polyfit2d(self.X, self.Y, self.Z, order = order)

        return m

    def removeFloor(self, order = 3, tresh_distance = .05):

        print('Removing floor...',end = '\r')
        # get bottom
        n, bins = np.histogram(self.Z, bins=int(np.max(self.Z) // .01))
        floor = bins[np.argmax(n)]
        floor_region = self.mask((self.Z > floor - .25) * (self.Z < floor + .25))

        m = floor_region.fitSurface(order=4)
        surface = polyval2d(np.column_stack((self.X, self.Y)), m)

        # get mask
        dist_mask = ((self.Z - surface) > tresh_distance)

        return self.mask(dist_mask)
        print('Floor Removed!')

    def slice(self, layer_size = 0.05):

        slices = []
        for z in np.arange(np.min(self.Z), np.max(self.Z), layer_size):
            mask = (self.Z > z)*(self.Z < (z + layer_size))
            slice = self.mask(mask)
            slices.append(slice)

        return slices

    def getFrontalArea(self, layer = 0.05, eps = 0.01, min_samples = 100, alpha = 5, ncores = 6):
        
        slices = self.slice(layer = layer)
        frontal_area_per_slice = np.zeros(len(slices))
        
        def getAlphashape(i, nclusters, xy, alpha = 5):
            #print(f'cluster progress: {i}/{nclusters}', end = '\r')
            return alphashape(xy, alpha = alpha)
            
        for i,slice in enumerate(slices):

            print(f'Calculating frontal area per slice: {i}/{len(slices)}')
            print(len(slice.points))

            if len(slice.points) > 3:
                clusters = slice.cluster(technique="DBSCAN", param={'eps': eps, 'min_samples': min_samples}, return_cluster_clouds=True,remove_noise=True)
                cluster_XYs = [np.column_stack((c.X, c.Z)) for c in clusters]

                if ncores == 1:
                    boxes = [alphashape(xy, alpha = alpha) for xy in cluster_XYs]
                else:
                    pool = PathosPool(ncores)
                    boxes = pool.map(getAlphashape,np.arange(len(cluster_XYs)),np.zeros(len(cluster_XYs), dtype = int) + len(cluster_XYs),cluster_XYs)
                    
                slice_frontal_area = np.sum([box.area for box in boxes])
                frontal_area_per_slice[i] = slice_frontal_area
            
            else:
                frontal_area_per_slice[i] = 0

        return frontal_area_per_slice

