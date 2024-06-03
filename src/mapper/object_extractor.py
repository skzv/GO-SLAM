import numpy as np
import geometry_utils


class ObjectExtractor:
    def __init__(self, c2ws, depths, camera):
        self.c2ws = c2ws
        self.depths = depths
        self.camera = camera
        self.H = camera.H
        self.W = camera.W

    def add_depth_to_objects(self, objects, index):
        image = np.zeros((self.H, self.W, 3))

        depth_map = self.depths[index]
        c2w = self.c2ws[index]

        for object in objects:
            mask = object["mask_in_image_coordinates"]

            world_coordinates = geometry_utils.pixel_to_world_coordinates(image, depth_map, self.camera.intrinsic_matrix, c2w)

            world_coordinates = geometry_utils.apply_mask_to_world_coordinates(world_coordinates, mask)
            object['world_coordinates_cm'] = self.quantize_world_coordinates_to_cm(world_coordinates)
            object['3d_box_world_coordinates'] = ObjectExtractor.min_max_3d_box(world_coordinates)

        return objects


    def min_max_3d_box(pcd):
        # pcd has format (N, 3) => (x, y, z)
        min_x = np.min(pcd[:, 0])
        max_x = np.max(pcd[:, 0])
        min_y = np.min(pcd[:, 1])
        max_y = np.max(pcd[:, 1])
        min_z = np.min(pcd[:, 2])
        max_z = np.max(pcd[:, 2])

        return {
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y,
            'min_z': min_z,
            'max_z': max_z
        }
    

    def group_objects_by_label(self, objects):
        objects_per_label = {}
        for object in objects:
            label = object['label']
            if label not in objects_per_label:
                objects_per_label[label] = []
            objects_per_label[label].append(object)
        return objects_per_label


    def combine_by_label(self, objects_per_label_1, objects_per_label_2):
        for label, objects in objects_per_label_2.items():
            if label not in objects_per_label_1:
                objects_per_label_1[label] = []
            objects_per_label_1[label].extend(objects)
        return objects_per_label_1
    
    def quantize_world_coordinates_to_cm(self, world_coordinates):
        # Quantize the world coordinates to integers
        return np.round(100 * world_coordinates).astype(int)

    def compute_overlap(self, world_coordinates1, world_coordinates2):
        # world_coordinates is (M, 3)
        # compute the overlap between two sets of world coordinates
        # the overlap is the number of points that are in both sets
        # divided by the total number of points in the two sets

        # Convert coordinates to integers and then to tuples to make them hashable
        set1 = set(map(tuple, world_coordinates1))
        set2 = set(map(tuple, world_coordinates2))
        
        overlap = len(set1.intersection(set2))
        total_unique_points = len(set1.union(set2))
    
        return overlap / total_unique_points
    
    def are_same_object(self, object1, object2):
        # two objects are considered the same if they have the same
        # label and enough overlapping world coordinates
        if not object1['label'] == object2['label']:
            return False
        overlap = self.compute_overlap(object1['world_coordinates_cm'], object2['world_coordinates_cm'])
        # print("Overlap: ", overlap)
        if overlap > 0.1:
            return True
        return False

    def merge_two_objects(self, object1, object2):
        # merge object2 into object1
        # Combine the world coordinates and remove duplicates

        combined_coords = np.vstack((object1['world_coordinates_cm'], object2['world_coordinates_cm']))
        combined_coords = np.unique(combined_coords, axis=0)
        object1['world_coordinates_cm'] = combined_coords
        # print(combined_coords.shape)

        object1['3d_box_world_coordinates'] = ObjectExtractor.min_max_3d_box(object1['world_coordinates_cm']/100)
        object1['merged_count'] = object1.get('merged_count', 1) +  object2.get('merged_count', 1)
        # print("Merged object with label", object1['label'], "from", object1['merged_count'], "frames")

    def merge_objects_with_same_labels(self, objects):
        merged = True
        while merged:
            merged = False
            i = 0
            while i < len(objects):
                j = i + 1
                while j < len(objects):
                    if self.are_same_object(objects[i], objects[j]):
                        self.merge_two_objects(objects[i], objects[j])
                        objects.pop(j)
                        merged = True
                    else:
                        j += 1
                i += 1
    
    def merge_all_objects(self, all_objects):
        for label, objects in all_objects.items():
            self.merge_objects_with_same_labels(objects)
        return all_objects
    
    def merge_objects(self, all_objects, new_objects):
        """
            Merge objects with overlapping boxes and count from how many frames they were extracted. 
            Discard objects that were extracted from less than 3 frames.
            """
        
        
        new_objects = self.group_objects_by_label(new_objects)

        if all_objects is None:
            all_objects = new_objects
            return all_objects
        all_objects = self.combine_by_label(all_objects, new_objects)
        all_objects = self.merge_all_objects(all_objects)

        return all_objects
    

    def remove_objects_below_count_threshold(self, all_objects, threshold):
        for label, objects in all_objects.items():
            objects = [object for object in objects if object.get('merged_count', 3) >= threshold]
            all_objects[label] = objects
        return all_objects

    def finish_merging(self, all_objects):
        all_objects = self.randomly_subsample_world_coordinates_for_all_objects(all_objects)
        all_objects = self.remove_objects_below_count_threshold(all_objects, 1)
        self.count_objects(all_objects)
        return all_objects

    def count_objects(self, all_objects):
        print("Number of objects per label:" , {label: len(objects) for label, objects in all_objects.items()})

    def randomly_subsample_world_coordinates_for_all_objects(self, all_objects):
        for label, objects in all_objects.items():
            for object in objects:
                sampling_rate = max(1/object.get('merged_count', 1), 0.1)
                world_coordinates = object['world_coordinates_cm']
                num_points = world_coordinates.shape[0]
                num_points_to_sample = int(num_points * sampling_rate)
                sampled_indices = np.random.choice(num_points, num_points_to_sample, replace=False)
                object['world_coordinates_cm'] = world_coordinates[sampled_indices]
        return all_objects

    def flatten_all_objects(self, all_objects):
        all_objects_flat = []
        for label, objects in all_objects.items():
            all_objects_flat.extend(objects)
        return all_objects_flat
