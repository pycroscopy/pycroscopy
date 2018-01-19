# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 17:03:29 2016

@author: Suhas Somnath
"""

from __future__ import division, print_function, absolute_import, unicode_literals
import numpy as np


# TODO: Test and debug node and clusterTree classes for agglomerative clustering etc

class Node(object):
    """
    Basic unit of a tree - a node. Keeps track of its value, labels, parent, children, level in the tree etc.
    """

    def __init__(self, name, value=None, parent=None, dist=0, labels=None, children=[], compute_mean=False,
                 verbose=False):
        """
        Parameters
        ----------
        name : (Optional) unsigned int
            ID of this node
        value : (Optional) 1D numpy array
            Response corresponding to this Node.
        parent : (Optional) unsigned int or Node object
            Parent for this Node.
        dist : (Optional) float
            Distance between the children nodes
        labels : (Optional) list or 1D numpy array of unsigned integers
            Positions / instances in a main dataset within this cluster
        children : (Optional) list of Node objects
            Children for this node
        compute_mean : (Optional) Boolean
            Whether or not to compute the value attribute from the provided children
        """
        self.children = children
        self.parent = parent
        self.name = name
        self.value = value
        self.dist = dist
        self.level = 0

        # Assign this node as the parent for all its children
        for child in children:
            child.parent = self

        # If labels were not provided (tree node), get from children
        if labels is None:
            temp_labels = []
            for child in self.children:
                if verbose:
                    print('Child #{} had the following labels:'.format(child.name))
                    print(child.labels)
                temp_labels.append(np.array(child.labels))
            if verbose:
                print('Labels (unsorted) derived from children for node #{}:'.format(name))
                print(temp_labels)
            self.labels = np.hstack(temp_labels)
            self.labels.sort()
        else:
            if verbose:
                print('Labels for leaf node #{}:'.format(name))
                print(labels)
            self.labels = np.array(labels, dtype=np.uint32)

        # Compute the level for this node along with the number of children below it
        if len(self.children) > 0:
            self.num_nodes = 0
            for child in self.children:
                self.num_nodes += child.num_nodes
                self.level = max(self.level, child.level)
            self.level += 1  # because this node has to be one higher level than its highest children
        else:
            self.num_nodes = 1
            if verbose:
                print('Parent node:', str(name), 'has', str(self.num_nodes), 'children')
        if all([len(self.children) > 0, value is None, compute_mean]):
            resp = []
            for child in children:
                if verbose:
                    print('       Child node', str(child.name), 'has', str(child.num_nodes), 'children')
                # primitive method of equal bias mean: resp.append(child.value)
                # weighted mean:
                resp.append(child.value * child.labels.size / self.labels.size)
                # self.value = np.mean(np.array(resp), axis=0)
            self.value = np.sum(np.array(resp), axis=0)

    def __str__(self):
        return '({}) --> {},{}'.format(self.name, str(self.children[0].name), str(self.children[1].name))


class ClusterTree(object):
    """
    Creates a tree representation from the provided linkage pairing. Useful for clustering
    """

    def __init__(self, linkage_pairing, labels, distances=None, centroids=None):
        """
        Parameters
        ----------
        linkage_pairing : 2D unsigned int numpy array or list
            Linkage pairing that describes a tree structure. The matrix should result in a single tree apex.
        labels : 1D unsigned int numpy array or list
            Labels assigned to each of the positions in the main dataset. Eg. Labels from clustering
        distances : (Optional) 1D numpy float array or list
            Distances between clusters
        centroids : (Optional) 2D numpy array
            Mean responses for each of the clusters. These will be propagated up
        """
        self.num_leaves = linkage_pairing.shape[0] + 1
        self.linkage = linkage_pairing
        self.centroids = centroids
        """ this list maintains pointers to the nodes pertaining to that cluster id for quick look-ups
        By default this lookup table just contains the number indices of these clusters.
        They will be replaced with node objects as and when the objects are created"""
        self.nodes = list()

        # now the labels is a giant list of labels assigned for each of the positions.
        self.labels = np.array(labels, dtype=np.uint32)
        """ the labels for the leaf nodes need to be calculated manually from the provided labels
        Populate the lowest level nodes / leaves first:"""
        for clust_id in range(self.num_leaves):
            which_pos = np.where(self.labels == clust_id)
            if centroids is not None:
                self.nodes.append(Node(clust_id, value=centroids[clust_id], labels=which_pos))
            else:
                self.nodes.append(Node(clust_id, labels=which_pos))

        for row in range(linkage_pairing.shape[0]):
            """print 'working on', linkage_pairing[row]
            we already have each of these children in our look-up table"""
            childs = []  # this is an empty list that will hold all the children corresponding to this node
            for col in range(linkage_pairing.shape[1]):
                """ look at each child in this row
                look up the node object corresponding to this label """
                childs.append(self.nodes[int(linkage_pairing[row, col])])
                # Now this row results in a new node. That is what we create here and assign the children to this node
            new_node = Node(row + self.num_leaves, children=childs, compute_mean=centroids is not None)
            # If distances are provided, add the distances attribute to this node.
            # This is the distance between the children
            if distances is not None:
                new_node.dist = distances[row]
                # add this node to the look-up table:
            self.nodes.append(new_node)

        self.tree = self.nodes[-1]

    def __str__(self):
        """
        Overrides the to string representation. Prints the names of the node and its children.
        Not very useful for large trees

        Returns
        --------
        String representation of the tree structure
        """
        return str(self.tree)
